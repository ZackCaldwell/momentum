import os
import json
import pandas as pd
import numpy as np
import datetime
import time
import logging
import requests
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#####################################################################
# Logging Configuration
#####################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#####################################################################
# Configuration
#####################################################################

# Alpaca API credentials and endpoint
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Change to paper URL for testing

# Trading Parameters
LOOKBACK_PERIOD = 42
REBALANCE_FREQ = 'M'  # 'M' for Monthly, 'W' for Weekly
NUM_STOCKS = 10
INITIAL_CASH = 20000
MAX_POSITION_SIZE = 0.1  # 10% of portfolio per position

# Universe Selection Parameters
UNIVERSE_SIZE = 200
MIN_AVG_VOL = 5_000_000
MIN_PRICE = 20.0

# Trading Costs and Slippage
COMMISSION_PER_TRADE = 1.0
SLIPPAGE_BPS = 10

# Risk Management Parameters
STOP_LOSS_FACTOR = 0.9
TAKE_PROFIT_FACTOR = 1.5  # Sell half if up > 50%
MIN_HOLD_DAYS = 30  # Minimum holding period in days

# Market Regime Symbol
MARKET_REGIME_SYMBOL = 'SPY'

# State Persistence File
STATE_FILE = 'trading_state.json'

# Email Configuration for Alerts
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
EMAIL_FROM = os.environ.get("EMAIL_FROM")
EMAIL_TO = os.environ.get("EMAIL_TO")

#####################################################################
# Initialize Alpaca API
#####################################################################

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

#####################################################################
# State Management Functions
#####################################################################

def load_state():
    """Load trading state from a JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            # Convert string dates back to datetime objects with timezone
            state['position_entry_dates'] = {k: pd.to_datetime(v) for k, v in state.get('position_entry_dates', {}).items()}
            return state
    return {"entry_prices": {}, "position_entry_dates": {}}

def save_state(state):
    """Save trading state to a JSON file."""
    state_to_save = state.copy()
    # Convert datetime objects to ISO format strings
    state_to_save['position_entry_dates'] = {k: v.isoformat() for k, v in state.get('position_entry_dates', {}).items()}
    with open(STATE_FILE, 'w') as f:
        json.dump(state_to_save, f, indent=4)

# Initialize state
state = load_state()

#####################################################################
# Email Notification Function
#####################################################################

def send_email(subject, body):
    """Send an email notification."""
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        logger.info("Sent notification email.")
    except Exception as e:
        logger.error(f"Error sending email: {e}")

#####################################################################
# Helper Functions
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
            bars = api.get_bars(chunk, timeframe, start=start_str, end=end_str, adjustment='raw').df
        except Exception as e:
            logger.error(f"Error fetching bars for chunk {idx}: {e}")
            continue

        if bars is None or bars.empty:
            logger.debug(f"No data returned for chunk {idx}")
            continue

        if 'symbol' not in bars.index.names:
            if 'symbol' in bars.columns:
                if 'timestamp' not in bars.columns:
                    if 'timestamp' in bars.index.names:
                        bars = bars.reset_index()
                if 'symbol' in bars.columns and 'timestamp' in bars.columns:
                    bars = bars.set_index(['symbol', 'timestamp']).sort_index()
                else:
                    if len(chunk) == 1:
                        sym = chunk[0]
                        if bars.index.name == 'timestamp':
                            bars['symbol'] = sym
                            bars = bars.reset_index().set_index(['symbol', 'timestamp']).sort_index()
                        else:
                            logger.warning("No suitable timestamp column found for single-symbol fallback.")
                            continue
                    else:
                        logger.warning(
                            "Multiple symbols requested but no 'symbol' and 'timestamp' columns after reset.")
                        continue

        if 'symbol' not in bars.index.names or 'timestamp' not in bars.index.names:
            logger.warning("Unable to set multi-index. Data may not be usable.")
            continue

        all_data.append(bars)

    if not all_data:
        logger.debug("No bars data collected.")
        return pd.DataFrame()

    data = pd.concat(all_data)
    data = data.sort_index()
    return data

def get_universe(api, start_date, end_date, universe_size=UNIVERSE_SIZE, min_avg_vol=MIN_AVG_VOL, min_price=MIN_PRICE):
    logger.debug("Fetching active assets from Alpaca.")
    try:
        active_assets = [a for a in api.list_assets(status='active') if a.tradable and a.exchange in ['NYSE', 'NASDAQ']]
    except Exception as e:
        logger.error(f"Error fetching active assets: {e}")
        return []

    symbols = [a.symbol for a in active_assets if a.easy_to_borrow and a.shortable]

    end_dt = pd.Timestamp(end_date, tz='UTC')
    start_dt = end_dt - pd.Timedelta(days=30)
    start_universe_str = start_dt.strftime('%Y-%m-%d')
    end_universe_str = end_dt.strftime('%Y-%m-%d')

    barset = get_bars(api, symbols, start_universe_str, end_universe_str, timeframe='1Day')
    if barset.empty:
        return []

    avg_vol = barset.groupby(level=0)['volume'].mean()
    liquid_symbols = avg_vol[avg_vol > min_avg_vol].index.tolist()
    if len(liquid_symbols) == 0:
        return []

    last_day_data = barset.groupby(level=0).tail(1)
    last_day_close = last_day_data['close']
    viable = last_day_close[last_day_close > min_price].index.unique()
    viable_symbols = viable.get_level_values(0).unique()
    liquid_symbols = list(set(liquid_symbols).intersection(set(viable_symbols)))
    if len(liquid_symbols) == 0:
        return []

    avg_vol_filtered = avg_vol.loc[liquid_symbols].sort_values(ascending=False)
    selected = avg_vol_filtered.head(universe_size).index.tolist()
    return [str(sym) for sym in selected]

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
    returns = prices[selected_stocks].pct_change().dropna()
    recent_returns = returns.tail(lookback)
    vol = recent_returns.std() * np.sqrt(252)
    vol = vol.replace({0: np.nan}).dropna()
    if vol.empty:
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
# Live Trading Functions
#####################################################################

def get_current_positions():
    """Fetch current positions from Alpaca."""
    positions = {}
    try:
        current_positions = api.list_positions()
        for pos in current_positions:
            positions[pos.symbol] = int(float(pos.qty))
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        send_email("Trading Bot Error", f"Error fetching positions: {e}")
    return positions

def get_account_cash():
    """Fetch available cash from Alpaca account."""
    try:
        account = api.get_account()
        cash = float(account.cash)
        return cash
    except Exception as e:
        logger.error(f"Error fetching account cash: {e}")
        send_email("Trading Bot Error", f"Error fetching account cash: {e}")
        return 0.0

def place_order(symbol, qty, side, order_type='market', time_in_force='day'):
    """Place an order with Alpaca."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        logger.info(f"Placed {side} order for {qty} shares of {symbol}. Order ID: {order.id}")
        send_email("Order Placed", f"Placed {side} order for {qty} shares of {symbol}. Order ID: {order.id}")
        return order.id
    except Exception as e:
        logger.error(f"Error placing order for {symbol}: {e}")
        send_email("Order Placement Failed", f"Failed to place {side} order for {qty} shares of {symbol}. Error: {e}")
        return None

def place_stop_loss_order(symbol, qty, stop_price):
    """Place a stop-loss order."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='stop',
            stop_price=stop_price,
            time_in_force='gtc'
        )
        logger.info(f"Placed stop-loss order for {qty} shares of {symbol} at {stop_price}. Order ID: {order.id}")
        send_email("Stop-Loss Order Placed", f"Placed stop-loss order for {qty} shares of {symbol} at {stop_price}. Order ID: {order.id}")
    except Exception as e:
        logger.error(f"Error placing stop-loss for {symbol}: {e}")
        send_email("Stop-Loss Order Failed", f"Failed to place stop-loss order for {qty} shares of {symbol}. Error: {e}")

def place_take_profit_order(symbol, qty, limit_price):
    """Place a take-profit order."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='limit',
            limit_price=limit_price,
            time_in_force='gtc'
        )
        logger.info(f"Placed take-profit order for {qty} shares of {symbol} at {limit_price}. Order ID: {order.id}")
        send_email("Take-Profit Order Placed", f"Placed take-profit order for {qty} shares of {symbol} at {limit_price}. Order ID: {order.id}")
    except Exception as e:
        logger.error(f"Error placing take-profit for {symbol}: {e}")
        send_email("Take-Profit Order Failed", f"Failed to place take-profit order for {qty} shares of {symbol}. Error: {e}")

def rebalance(api, current_date, state):
    """Rebalance the portfolio based on the strategy."""
    logger.info(f"Rebalancing portfolio on {current_date.date()}.")
    send_email("Rebalancing Portfolio", f"Rebalancing portfolio on {current_date.date()}.")

    # Fetch universe and prices
    lookback_days = LOOKBACK_PERIOD * 2  # Adjust as needed
    start_date = current_date - pd.Timedelta(days=lookback_days)
    end_date = current_date

    universe = get_universe(api, start_date, end_date, universe_size=UNIVERSE_SIZE, min_avg_vol=MIN_AVG_VOL, min_price=MIN_PRICE)
    if not universe:
        logger.warning("No universe selected.")
        send_email("Rebalance Warning", "No universe selected during rebalance.")
        return

    prices_data = get_bars(api, universe, start_date, end_date, timeframe='1Day')
    if prices_data.empty:
        logger.warning("No price data fetched.")
        send_email("Rebalance Warning", "No price data fetched during rebalance.")
        return

    close_prices = prices_data['close'].unstack(level=0)
    close_prices = close_prices.dropna(axis=1, how='all')

    selected = select_stocks(close_prices, n=NUM_STOCKS, lookback=LOOKBACK_PERIOD)
    if not selected:
        logger.warning("No stocks selected by the strategy.")
        send_email("Rebalance Warning", "No stocks selected by the strategy during rebalance.")
        return

    # Market regime filter
    if MARKET_REGIME_SYMBOL in close_prices.columns:
        spy_prices = close_prices[MARKET_REGIME_SYMBOL].dropna()
        if len(spy_prices) > 200:
            spy_sma_200 = spy_prices.rolling(200).mean().iloc[-1]
            if spy_prices.iloc[-1] < spy_sma_200:
                logger.info("Market regime negative, halving target weights.")
                send_email("Market Regime Alert", "SPY is below its 200-day SMA. Halving target weights.")
                selected = selected[:max(1, int(len(selected) / 2))]

    if selected:
        target_weights = volatility_scale(close_prices, selected, lookback=LOOKBACK_PERIOD)
    else:
        target_weights = {}

    # Fetch current positions and account cash
    positions = get_current_positions()
    cash = get_account_cash()
    pos_value = sum([positions.get(s, 0) * close_prices[s].iloc[-1] for s in positions if s in close_prices.columns])
    portfolio_value = cash + pos_value

    # Determine trades
    trades = rebalance_portfolio(
        current_positions=positions,
        target_weights=target_weights,
        current_prices=close_prices.iloc[-1],
        portfolio_value=portfolio_value,
        entry_prices=state['entry_prices'],
        position_entry_dates=state['position_entry_dates'],
        current_date=current_date
    )

    # Execute trades
    for sym, qty in trades.items():
        if qty > 0:
            order_id = place_order(sym, qty, 'buy')
            if order_id:
                # Optionally, place stop-loss and take-profit orders
                entry_price = close_prices[sym].iloc[-1]
                stop_price = entry_price * STOP_LOSS_FACTOR
                limit_price = entry_price * TAKE_PROFIT_FACTOR
                place_stop_loss_order(sym, qty, stop_price)
                place_take_profit_order(sym, int(qty / 2), limit_price)  # Sell half for take-profit
                # Update state
                state['entry_prices'][sym] = entry_price
                state['position_entry_dates'][sym] = current_date.isoformat()
        elif qty < 0:
            order_id = place_order(sym, abs(qty), 'sell')
            if order_id:
                # Update state
                if sym in state['entry_prices']:
                    del state['entry_prices'][sym]
                if sym in state['position_entry_dates']:
                    del state['position_entry_dates'][sym]

    # Save updated state
    save_state(state)

#####################################################################
# Scheduler Function
#####################################################################

def is_rebalance_day(current_date, freq=REBALANCE_FREQ):
    """Determine if today is a rebalance day based on frequency."""
    if freq == 'M':
        # Rebalance on the first trading day of the month
        return current_date.day <= 7  # Assuming rebalance within first week
    elif freq == 'W':
        # Rebalance on Mondays
        return current_date.weekday() == 0
    # Add other frequencies as needed
    return False

#####################################################################
# Main Live Trading Loop
#####################################################################

def main():
    logger.info("Starting live trading bot.")
    send_email("Trading Bot Started", "The live trading bot has started.")

    while True:
        try:
            # Define the timezone for Eastern Time (ET)
            eastern = datetime.timezone(datetime.timedelta(hours=-5))  # Note: This does not account for daylight saving

            # Get current time in US Eastern Time
            now = datetime.datetime.now(tz=eastern)
            current_date = pd.Timestamp(now)  # Remove the tz parameter since 'now' is already tz-aware

            logger.info(f"Current date: {current_date.date()}")

            # Check if today is a rebalance day
            if is_rebalance_day(current_date, REBALANCE_FREQ):
                rebalance(api, current_date, state)

            # Sleep until next day at 6 PM ET to ensure all market data is available
            # Calculate seconds until next day at 6 PM ET
            next_run = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(18, 0, 0))
            next_run = next_run.replace(tzinfo=eastern)  # Ensure next_run is timezone-aware
            seconds_until_next_run = (next_run - now).total_seconds()
            if seconds_until_next_run < 0:
                seconds_until_next_run += 86400  # Add a day in seconds

            logger.info(f"Sleeping for {seconds_until_next_run} seconds until next check.")
            time.sleep(seconds_until_next_run)

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}")
            send_email("Trading Bot Error", f"An error occurred in the main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main()
