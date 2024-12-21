import os
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#####################################################################
# Logging Configuration
#####################################################################
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#####################################################################
# Configuration
#####################################################################

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

LOOKBACK_PERIOD = 42
REBALANCE_FREQ = 'M'
NUM_STOCKS = 10
INITIAL_CASH = 20000
MAX_POSITION_SIZE = 0.1

START_DATE = "2022-10-01"
END_DATE = "2024-12-18"

UNIVERSE_SIZE = 200
MIN_AVG_VOL = 5_000_000
MIN_PRICE = 20.0

COMMISSION_PER_TRADE = 1.0
SLIPPAGE_BPS = 10
STOP_LOSS_FACTOR = 0.9
TAKE_PROFIT_FACTOR = 1.5  # Sell half if up > 50%
MIN_HOLD_DAYS = 30  # Minimum holding period in days

MARKET_REGIME_SYMBOL = 'SPY'

#####################################################################
# Data Fetching and Universe Selection
#####################################################################

def get_bars(api, symbols, start_date, end_date, timeframe='1Day'):
    symbols = [str(s) for s in symbols if isinstance(s, str) or not pd.isnull(s)]
    logger.debug(f"Requesting bars for {len(symbols)} symbols from {start_date} to {end_date}.")
    if len(symbols) == 0:
        logger.debug("No symbols provided to get_bars.")
        return pd.DataFrame()

    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

    all_data = []
    chunk_size = 100
    symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    for idx, chunk in enumerate(symbol_chunks, start=1):
        if not chunk:
            continue

        logger.debug(f"Fetching bars for chunk {idx}/{len(symbol_chunks)}: {chunk}")
        try:
            bars = api.get_bars(chunk, timeframe, start=start_str, end=end_str, adjustment='split')
            df = bars.df

            if df is None or df.empty:
                logger.debug(f"No data returned for chunk {idx}")
                continue

            if 'symbol' not in df.index.names:
                # Try to fix the index if possible
                if 'symbol' in df.columns:
                    if 'timestamp' not in df.columns:
                        if 'timestamp' in df.index.names:
                            df = df.reset_index()
                    if 'symbol' in df.columns and 'timestamp' in df.columns:
                        df = df.set_index(['symbol', 'timestamp']).sort_index()
                    else:
                        if len(chunk) == 1:
                            sym = chunk[0]
                            if df.index.name == 'timestamp':
                                df['symbol'] = sym
                                df = df.reset_index().set_index(['symbol', 'timestamp']).sort_index()
                            else:
                                logger.warning("No suitable timestamp column found for single-symbol fallback.")
                                continue
                        else:
                            logger.warning("Multiple symbols requested but no 'symbol' and 'timestamp' columns after reset.")
                            continue

            if 'symbol' not in df.index.names or 'timestamp' not in df.index.names:
                logger.warning("Unable to set multi-index. Data may not be usable.")
                continue

            all_data.append(df)

        except Exception as e:
            logger.error(f"Error fetching data for chunk {idx}: {e}")
            continue

    if not all_data:
        logger.debug("No bars data collected.")
        return pd.DataFrame()

    data = pd.concat(all_data)
    data = data.sort_index()
    logger.debug(f"Total data fetched: {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def get_universe(api, start_date, end_date, universe_size=UNIVERSE_SIZE, min_avg_vol=MIN_AVG_VOL, min_price=MIN_PRICE):
    logger.debug("Fetching active assets from Alpaca.")
    active_assets = [a for a in api.list_assets(status='active') if a.tradable and a.exchange in ['NYSE', 'NASDAQ']]
    symbols = [a.symbol for a in active_assets if a.easy_to_borrow and a.shortable]

    end_dt = pd.Timestamp(end_date, tz='UTC')
    start_dt = end_dt - pd.Timedelta(days=30)
    start_universe_str = start_dt.strftime('%Y-%m-%d')
    end_universe_str = end_dt.strftime('%Y-%m-%d')

    logger.debug(f"Fetching bars for universe selection from {start_universe_str} to {end_universe_str}.")
    barset = get_bars(api, symbols, start_universe_str, end_universe_str, timeframe='1Day')
    if barset.empty:
        logger.warning("No bars data fetched for universe selection.")
        return []

    avg_vol = barset.groupby(level=0)['volume'].mean()
    liquid_symbols = avg_vol[avg_vol > min_avg_vol].index.tolist()
    if len(liquid_symbols) == 0:
        logger.warning("No liquid symbols found based on average volume criteria.")
        return []

    last_day_data = barset.groupby(level=0).tail(1)
    last_day_close = last_day_data['close']
    viable = last_day_close[last_day_close > min_price].index.unique()
    viable_symbols = viable.get_level_values(0).unique()
    liquid_symbols = list(set(liquid_symbols).intersection(set(viable_symbols)))
    if len(liquid_symbols) == 0:
        logger.warning("No symbols passed the price filter.")
        return []

    avg_vol_filtered = avg_vol.loc[liquid_symbols].sort_values(ascending=False)
    selected = avg_vol_filtered.head(universe_size).index.tolist()
    logger.debug(f"Selected {len(selected)} symbols for the universe based on volume and price criteria.")
    return [str(sym) for sym in selected]


#####################################################################
# Strategy Logic
#####################################################################

def calculate_momentum(prices, lookback=LOOKBACK_PERIOD):
    """
    Calculate momentum based on the average of past N daily returns.

    Parameters:
    - prices (pd.DataFrame): DataFrame of price data with dates as index and symbols as columns.
    - lookback (int): Number of periods to look back.

    Returns:
    - momentum (pd.Series): Normalized momentum scores for each symbol.
    """
    # Calculate daily returns
    daily_returns = prices.pct_change()

    # Calculate the average return over the lookback period
    avg_momentum = daily_returns.rolling(window=lookback).mean().iloc[-1]

    # Drop symbols with insufficient data
    avg_momentum = avg_momentum.dropna()

    # Rank the momentum scores in descending order (higher momentum gets a higher rank)
    momentum_rank = avg_momentum.rank(ascending=False, na_option='bottom')

    # Normalize the momentum ranks to have values between 0 and 1
    momentum_score = momentum_rank / momentum_rank.max()

    return momentum_score


def select_stocks(prices, n=NUM_STOCKS, lookback=LOOKBACK_PERIOD):
    """
    Select top N stocks based on momentum scores.

    Parameters:
    - prices (pd.DataFrame): DataFrame of price data with dates as index and symbols as columns.
    - n (int): Number of top stocks to select.
    - lookback (int): Number of periods to look back for momentum calculation.

    Returns:
    - selected (list): List of selected stock symbols.
    """
    mom = calculate_momentum(prices, lookback=lookback)
    mom = mom.dropna()

    if mom.empty:
        logger.debug("Momentum scores are empty after dropping NaNs.")
        return []

    # Sort the momentum scores in descending order and select the top N
    selected = mom.sort_values(ascending=False).head(n).index.tolist()

    logger.debug(f"Selected stocks based on momentum: {selected}")

    return selected


def volatility_scale(prices, selected_stocks, lookback=LOOKBACK_PERIOD):
    """
    Scale the target weights based on inverse volatility.

    Parameters:
    - prices (pd.DataFrame): Historical price data.
    - selected_stocks (list): List of selected stock symbols.
    - lookback (int): Number of periods to look back for volatility calculation.

    Returns:
    - weights (dict): Dictionary of target weights for each selected stock.
    """
    returns = prices[selected_stocks].pct_change().dropna()
    recent_returns = returns.tail(lookback)
    vol = recent_returns.std() * np.sqrt(252)
    vol = vol.replace({0: np.nan}).dropna()
    if vol.empty:
        # If volatility is zero for all, assign equal weights
        return {s: 1.0 / len(selected_stocks) for s in selected_stocks}
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    for s in selected_stocks:
        if s not in weights.index:
            weights[s] = 1.0 / len(selected_stocks)
    return weights.to_dict()


def rebalance_portfolio(current_positions, target_weights, current_prices, portfolio_value, entry_prices,
                        position_entry_dates, current_date):
    logger.debug("Rebalancing portfolio.")
    trades = {}

    sum_weights = sum(target_weights.values())
    if sum_weights > 1:
        factor = 1.0 / sum_weights
        target_weights = {k: v * factor for k, v in target_weights.items()}
    target_weights = {k: min(v, MAX_POSITION_SIZE) for k, v in target_weights.items()}

    # Check stop-loss and holding period (no selling if held < MIN_HOLD_DAYS)
    for sym in list(current_positions.keys()):
        if sym in entry_prices:
            current_price = current_prices.get(sym, np.nan)
            entry_price = entry_prices[sym]
            held_days = (current_date - position_entry_dates.get(sym, current_date)).days
            if not np.isnan(current_price):
                # Stop-loss
                if current_price < entry_price * STOP_LOSS_FACTOR and held_days >= MIN_HOLD_DAYS:
                    shares_to_trade = -current_positions[sym]
                    if shares_to_trade != 0:
                        trades[sym] = shares_to_trade
                        if sym in target_weights:
                            del target_weights[sym]
                # Take-profit: if up > 50%, sell half (also only if minimum hold period passed)
                elif current_price > entry_price * TAKE_PROFIT_FACTOR and held_days >= MIN_HOLD_DAYS:
                    # Sell half the position
                    shares_to_trade = -int(current_positions[sym] / 2)
                    if shares_to_trade != 0:
                        trades[sym] = trades.get(sym, 0) + shares_to_trade

    # Allocate per target_weights
    for sym, w in target_weights.items():
        price = current_prices.get(sym, np.nan)
        if np.isnan(price):
            continue
        target_value = w * portfolio_value
        current_shares = current_positions.get(sym, 0)
        current_val = current_shares * price
        diff_value = target_value - current_val
        if price > 0:
            shares_to_trade = int(diff_value / price)
            if shares_to_trade != 0:
                # Check if we are allowed to sell (if shares_to_trade < 0) only if min hold days passed
                if shares_to_trade < 0 and sym in position_entry_dates:
                    held_days = (current_date - position_entry_dates[sym]).days
                    if held_days < MIN_HOLD_DAYS:
                        # Skip selling this symbol due to min hold period
                        continue
                trades[sym] = trades.get(sym, 0) + shares_to_trade

    # Close positions not in target_weights if min hold period allows
    for sym in list(current_positions.keys()):
        if sym not in target_weights and sym not in trades:
            # Sell only if min hold period passed
            held_days = (current_date - position_entry_dates.get(sym, current_date)).days
            if held_days >= MIN_HOLD_DAYS:
                shares_to_trade = -current_positions[sym]
                if shares_to_trade != 0:
                    trades[sym] = shares_to_trade

    return trades


def get_rebalance_dates(close_prices, freq=REBALANCE_FREQ):
    """
    Determines the rebalance dates based on the specified frequency.

    Parameters:
    - close_prices (pd.DataFrame): DataFrame containing close prices with a DateTime index.
    - freq (str): Rebalance frequency code. Supported values:
        'W'  : Weekly (default day: Monday)
        'MS' : Monthly Start
        'QS' : Quarterly Start
        'AS' : Annually Start
        'YS' : Another Annually Start alias

    Returns:
    - rebal_dates (list of pd.Timestamp): List of dates to rebalance the portfolio.
    """
    # Map common frequency codes to pandas period frequencies
    freq_map = {
        'W': 'W-MON',  # Weekly on Monday
        'MS': 'M',      # Monthly Start
        'QS': 'Q',      # Quarterly Start
        'AS': 'A',      # Annually Start
        'YS': 'A'       # Another Annually Start alias
    }

    # Get the corresponding pandas frequency string, default to monthly if unknown
    period_freq = freq_map.get(freq, 'M')  # Default to monthly if unknown

    # Convert the index to the specified period frequency
    periods = close_prices.index.to_period(period_freq).unique()

    # Determine the first trading day of each period
    rebal_dates = []
    for p in periods:
        # Filter the close_prices for the current period
        period_data = close_prices[close_prices.index.to_period(period_freq) == p]
        if not period_data.empty:
            # Append the first date of the period as the rebalance date
            rebal_dates.append(period_data.index[0])

    return rebal_dates


#####################################################################
# Backtesting Engine
#####################################################################

def backtest(api, start_date=START_DATE, end_date=END_DATE):
    logger.info(f"Starting backtest from {start_date} to {end_date}.")
    universe = get_universe(api, start_date, end_date, universe_size=UNIVERSE_SIZE, min_avg_vol=MIN_AVG_VOL,
                            min_price=MIN_PRICE)
    if not universe:
        logger.warning("No universe selected.")
        return None, [], []

    logger.info(f"Universe selected: {len(universe)} symbols.")

    prices_data = get_bars(api, universe, start_date, end_date, timeframe='1Day')
    if prices_data.empty:
        logger.warning("No data fetched for selected universe.")
        return None, [], []

    # Attempt to extract 'close' prices
    try:
        close_prices = prices_data['close'].unstack(level=0)
    except KeyError:
        logger.error("'close' column not found in prices_data.")
        return None, [], []

    close_prices = close_prices.dropna(axis=1, how='all')

    # **Fix Applied Here: Correctly reset MultiIndex if present**
    if isinstance(close_prices.index, pd.MultiIndex):
        # Correctly check if both 'symbol' and 'timestamp' are in the index names
        if 'symbol' in close_prices.index.names and 'timestamp' in close_prices.index.names:
            close_prices = close_prices.reset_index(level='symbol', drop=True)
            logger.debug("Reset MultiIndex to have timestamp as index.")
        else:
            logger.warning("MultiIndex does not contain both 'symbol' and 'timestamp'. Unable to reset index properly.")
            return None, [], []

    # Verify that the index is now single-level
    if isinstance(close_prices.index, pd.MultiIndex):
        logger.error("Failed to reset MultiIndex. 'close_prices' still has a MultiIndex.")
        return None, [], []
    else:
        logger.debug("close_prices has a single-level DateTimeIndex.")
        logger.debug(f"close_prices index type: {type(close_prices.index)}")
        logger.debug(f"close_prices index name: {close_prices.index.name}")
        assert isinstance(close_prices.index, pd.DatetimeIndex), "Index is not a DateTimeIndex."

    close_prices = close_prices.dropna(axis=1, how='all')

    # Log the structure of close_prices
    logger.debug(f"close_prices type: {type(close_prices)}")
    logger.debug(f"close_prices columns: {close_prices.columns.tolist()}")
    logger.debug(f"close_prices head:\n{close_prices.head()}")

    universe = close_prices.columns.tolist()
    logger.info(f"{len(universe)} symbols remain after filtering.")

    if len(universe) == 0:
        logger.warning("No symbols remain after filtering.")
        return None, [], []

    rebal_dates = get_rebalance_dates(close_prices, REBALANCE_FREQ)
    logger.debug(f"Rebalance dates: {rebal_dates}")

    cash = INITIAL_CASH
    positions = {}
    portfolio_history = []
    entry_prices = {}  # Average cost per share for each symbol
    position_entry_dates = {}

    # Initialize trade logs
    trade_log = []  # For position entry/exit summaries
    execution_log = []  # For detailed trade executions

    logger.info("Starting portfolio simulation.")
    for i in range(len(close_prices)):
        current_date = close_prices.index[i]
        current_prices = close_prices.iloc[i]
        pos_value = sum([positions.get(s, 0) * current_prices.get(s, np.nan)
                         for s in positions if s in current_prices.index])
        total_equity = cash + pos_value

        if current_date in rebal_dates:
            logger.debug(f"Rebalance day: {current_date}")
            hist_window = close_prices.iloc[:i]
            if len(hist_window) < LOOKBACK_PERIOD:
                logger.debug("Not enough history for momentum. Skipping.")
            else:
                selected = select_stocks(hist_window, n=NUM_STOCKS, lookback=LOOKBACK_PERIOD)

                # Market regime filter
                if MARKET_REGIME_SYMBOL in hist_window.columns:
                    spy_prices = hist_window[MARKET_REGIME_SYMBOL].dropna()
                    if len(spy_prices) > 200:
                        spy_sma_200 = spy_prices.rolling(200).mean().iloc[-1]
                        if spy_prices.iloc[-1] < spy_sma_200:
                            logger.debug("Market regime negative, halving target weights.")
                            selected = selected[:max(1, int(len(selected) / 2))]

                if len(selected) > 0:
                    target_weights = volatility_scale(hist_window, selected, lookback=LOOKBACK_PERIOD)
                else:
                    target_weights = {}

                trades = rebalance_portfolio(
                    positions, target_weights, current_prices, total_equity, entry_prices,
                    position_entry_dates, current_date
                )

                for sym, shares_delta in trades.items():
                    if sym not in current_prices.index or np.isnan(current_prices[sym]):
                        logger.debug(f"No price data for {sym} today, skipping trade.")
                        continue

                    # Enhanced Logging: Log the current price before executing the trade
                    logger.debug(f"Executing trade for {sym} on {current_date}: Current Price = {current_prices[sym]}")

                    trade_price = current_prices[sym]
                    trade_price *= (1 + SLIPPAGE_BPS / 10000.0) if shares_delta > 0 else (1 - SLIPPAGE_BPS / 10000.0)

                    # Log the adjusted trade price
                    logger.debug(f"Adjusted Trade Price for {sym}: {trade_price}")

                    if shares_delta > 0:
                        # Buy
                        trade_cost = shares_delta * trade_price + COMMISSION_PER_TRADE
                        if cash - trade_cost < 0:
                            max_shares = int((cash - COMMISSION_PER_TRADE) / trade_price)
                            if max_shares <= 0:
                                logger.debug(f"Not enough cash for {sym}. Skipping.")
                                continue
                            trade_cost = max_shares * trade_price + COMMISSION_PER_TRADE
                            shares_delta = max_shares
                            logger.debug(f"Adjusted shares to buy for {sym}: {shares_delta}")

                        cash -= trade_cost
                        old_shares = positions.get(sym, 0)
                        new_shares = old_shares + shares_delta
                        positions[sym] = new_shares

                        # Update average entry price
                        if old_shares == 0:
                            entry_prices[sym] = trade_price
                            position_entry_dates[sym] = current_date
                        else:
                            # Weighted average entry price
                            entry_prices[sym] = ((entry_prices[sym] * old_shares) + (trade_price * shares_delta)) / new_shares

                        # Record Execution Log
                        execution_log.append({
                            'Date': current_date,
                            'Symbol': sym,
                            'Action': 'Buy',
                            'Shares': shares_delta,
                            'Price': trade_price,
                            'Commission': COMMISSION_PER_TRADE,
                            'Total Cost/Proceeds': shares_delta * trade_price + COMMISSION_PER_TRADE,
                            'Cash After Trade': cash,
                            'Portfolio Value After Trade': cash + sum(
                                [positions.get(s, 0) * current_prices.get(s, np.nan) for s in positions if s in current_prices.index])
                        })

                    elif shares_delta < 0:
                        # Sell
                        shares_to_sell = abs(shares_delta)
                        current_shares = positions.get(sym, 0)
                        if shares_to_sell > current_shares:
                            shares_to_sell = current_shares

                        # Calculate profit for the shares being sold
                        profit_per_share = trade_price - entry_prices.get(sym, 0)
                        realized_profit = profit_per_share * shares_to_sell

                        # Update positions and cash
                        trade_proceeds = shares_to_sell * trade_price - COMMISSION_PER_TRADE
                        cash += trade_proceeds
                        new_shares = current_shares - shares_to_sell
                        positions[sym] = new_shares

                        # Record Execution Log
                        execution_log.append({
                            'Date': current_date,
                            'Symbol': sym,
                            'Action': 'Sell',
                            'Shares': shares_to_sell,
                            'Price': trade_price,
                            'Commission': COMMISSION_PER_TRADE,
                            'Total Cost/Proceeds': shares_to_sell * trade_price - COMMISSION_PER_TRADE,
                            'Cash After Trade': cash,
                            'Portfolio Value After Trade': cash + sum(
                                [positions.get(s, 0) * current_prices.get(s, np.nan) for s in positions if s in current_prices.index])
                        })

                        # Record Trade Log for realized profit
                        trade_log.append({
                            'Symbol': sym,
                            'Entry Date': position_entry_dates.get(sym, np.nan),
                            'Exit Date': current_date,
                            'Entry Price': entry_prices.get(sym, 0),
                            'Exit Price': trade_price,
                            'Shares': shares_to_sell,
                            'Profit': realized_profit
                        })

                        if new_shares == 0:
                            del entry_prices[sym]
                            del position_entry_dates[sym]

                    logger.debug(f"Executed trade for {sym}: {shares_delta} shares at {trade_price}")

        portfolio_history.append([current_date, total_equity, cash, pos_value])

    # After loop ends, close any open positions at the last price
    logger.info("Closing any remaining open positions at the end of backtest period.")
    for sym, shares in positions.items():
        if shares != 0 and sym in close_prices.columns:
            logger.debug(f"Closing position for {sym}: {shares} shares at {close_prices[sym].iloc[-1]}")
            exit_price = close_prices[sym].iloc[-1]
            profit = (exit_price - entry_prices.get(sym, 0)) * shares
            trade_log.append({
                'Symbol': sym,
                'Entry Date': position_entry_dates.get(sym, np.nan),
                'Exit Date': close_prices.index[-1],
                'Entry Price': entry_prices.get(sym, 0),
                'Exit Price': exit_price,
                'Shares': shares,
                'Profit': profit
            })
            # Record Execution Log for closing the position
            execution_log.append({
                'Date': close_prices.index[-1],
                'Symbol': sym,
                'Action': 'Sell',
                'Shares': shares,
                'Price': exit_price,
                'Commission': COMMISSION_PER_TRADE,
                'Total Cost/Proceeds': shares * exit_price - COMMISSION_PER_TRADE,
                'Cash After Trade': cash + shares * exit_price - COMMISSION_PER_TRADE,
                'Portfolio Value After Trade': cash + shares * exit_price - COMMISSION_PER_TRADE
            })

    df_history = pd.DataFrame(portfolio_history, columns=['Date', 'Equity', 'Cash', 'Positions_Value'])
    df_history.set_index('Date', inplace=True)
    logger.info("Backtest complete.")

    return df_history, trade_log, execution_log


#####################################################################
# Performance Metrics
#####################################################################

def performance_metrics(df, trade_log, risk_free_rate=0.0):
    logger.debug("Calculating performance metrics.")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")  # Verify column names
    df = df.copy()
    df['returns'] = df['Equity'].pct_change().fillna(0)

    start_val = df['Equity'].iloc[0]
    end_val = df['Equity'].iloc[-1]
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25 if total_days > 0 else np.nan

    # Total Return
    total_return = (end_val / start_val) - 1 if start_val else np.nan

    # CAGR calculation
    cagr = (end_val / start_val) ** (1 / years) - 1 if years and years > 0 else np.nan

    # Sharpe Ratio
    daily_returns = df['returns']
    mean_daily_return = daily_returns.mean()
    daily_vol = daily_returns.std()
    annualized_return = (mean_daily_return - risk_free_rate / 252) * 252
    annualized_volatility = daily_vol * np.sqrt(252)
    sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

    # Sortino Ratio
    downside_returns = df['returns'][df['returns'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (annualized_return) / downside_vol if downside_vol != 0 else np.nan

    # Max Drawdown
    running_max = df['Equity'].cummax()
    drawdowns = df['Equity'] / running_max - 1
    max_dd = drawdowns.min()

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Maximum Drawdown Duration
    drawdown_periods = (drawdowns != 0).cumsum()
    drawdown_groups = drawdowns.groupby(drawdown_periods)
    max_dd_duration = 0
    for name, group in drawdown_groups:
        if group.iloc[0] < 0:
            duration = len(group)
            if duration > max_dd_duration:
                max_dd_duration = duration

    # Total number of trades
    total_trades = len(trade_log)

    # Win rate and profit factor
    if total_trades > 0:
        profitable_trades = [trade for trade in trade_log if trade['Profit'] > 0]
        win_rate = len(profitable_trades) / total_trades
        gross_profit = sum([trade['Profit'] for trade in profitable_trades])
        gross_loss = sum([-trade['Profit'] for trade in trade_log if trade['Profit'] < 0])
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
    else:
        win_rate = np.nan
        profit_factor = np.nan

    # Average Trade Profit/Loss
    if total_trades > 0:
        average_profit = sum([trade['Profit'] for trade in trade_log]) / total_trades
    else:
        average_profit = np.nan

    # Realized vs Portfolio Profit Validation
    total_realized_profit = sum(trade['Profit'] for trade in trade_log)
    total_portfolio_profit = end_val - start_val

    logger.debug(f"Total Realized Profit: {total_realized_profit}")
    logger.debug(f"Total Portfolio Profit: {total_portfolio_profit}")

    discrepancy = abs(total_realized_profit - total_portfolio_profit)
    if discrepancy > 1e-2:
        logger.warning(f"Discrepancy detected: Realized Profit ({total_realized_profit}) != Portfolio Profit ({total_portfolio_profit})")
    else:
        logger.info("Trade log and portfolio profit are consistent.")

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': annualized_volatility,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Max Drawdown Duration (days)': max_dd_duration,
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Average Trade Profit/Loss': average_profit  # Optional
    }



#####################################################################
# Main Execution
#####################################################################

if __name__ == "__main__":
    logger.info("Initializing Alpaca API.")
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca API: {e}")
        raise

    # Perform the backtest
    results, trades, executions = backtest(api, START_DATE, END_DATE)

    if results is not None and not results.empty:
        # Calculate performance metrics
        metrics = performance_metrics(results, trades)
        print("Backtest Results:")
        print(results.tail())

        print("\nPerformance Metrics:")
        for k, v in metrics.items():
            if pd.notnull(v):
                if k in ['CAGR', 'Max Drawdown', 'Total Return']:
                    # These metrics make sense as percentages
                    print(f"{k}: {v:.2%}")
                elif k in ['Sharpe', 'Sortino', 'Calmar Ratio', 'Profit Factor']:
                    # Ratios should be printed as normal floats
                    print(f"{k}: {v:.2f}")
                elif k == 'Max Drawdown Duration (days)':
                    print(f"{k}: {v} days")
                elif k == 'Win Rate':
                    print(f"{k}: {v:.2%}")
                elif k == 'Average Trade Profit/Loss':
                    print(f"{k}: {v:.2f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: N/A")

        # Plot equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['Equity'], label='Equity Curve', color='blue')
        plt.title('Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Equity ($)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Detailed Execution Log
        if executions:
            print("\nDetailed Execution Log:")
            execution_df = pd.DataFrame(executions)
            pd.set_option('display.max_rows', None)
            print(execution_df)
            pd.set_option('display.max_rows', 10)  # Reset to default

            # Save Execution Log to CSV
            execution_df.to_csv('execution_log.csv', index=False)
            print("\nExecution log saved to 'execution_log.csv'.")

        # Trade Summary Log
        if trades:
            print("\nTrade Summary Log:")
            trade_df = pd.DataFrame(trades)
            print(trade_df)
            trade_df.to_csv('trade_summary_log.csv', index=False)
            print("\nTrade summary log saved to 'trade_summary_log.csv'.")
    else:
        logger.warning("No results. Check data retrieval or parameters.")
        print("No results. Check data retrieval or parameters.")
