import os
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import logging
import pickle

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

LOOKBACK_PERIOD = 30
REBALANCE_FREQ = 'W'
NUM_STOCKS = 10
INITIAL_CASH = 20000
MAX_POSITION_SIZE = 0.1

START_DATE = "2023-10-01"
END_DATE = "2024-12-18"

UNIVERSE_SIZE = 250
MIN_AVG_VOL = 6_000_000
MIN_PRICE = 20.0

COMMISSION_PER_TRADE = 1.0
SLIPPAGE_BPS = 10
STOP_LOSS_FACTOR = 0.9
TAKE_PROFIT_FACTOR = 1.5  # Sell half if up > 50%
MIN_HOLD_DAYS = 30  # Minimum holding period in days

MARKET_REGIME_SYMBOL = 'SPY'

#####################################################################
# Caching Configuration
#####################################################################

CACHE_DIR = "price_data_cache"


def ensure_cache_dir():
    """
    Ensures that the cache directory exists.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.debug(f"Created cache directory at {CACHE_DIR}.")


def get_cache_path(symbol, start_date, end_date, timeframe='1Day'):
    """
    Generates a cache file path for a given symbol and date range.

    Parameters:
    - symbol (str): Stock symbol.
    - start_date (str): Start date in 'YYYY-MM-DD'.
    - end_date (str): End date in 'YYYY-MM-DD'.
    - timeframe (str): Timeframe for the bars.

    Returns:
    - path (str): Path to the cache file.
    """
    filename = f"{symbol}_{start_date}_{end_date}_{timeframe}.pkl"
    return os.path.join(CACHE_DIR, filename)


def load_cached_bars(symbol, start_date, end_date, timeframe='1Day'):
    """
    Loads cached bars for a symbol if available.

    Parameters:
    - symbol (str): Stock symbol.
    - start_date (str): Start date in 'YYYY-MM-DD'.
    - end_date (str): End date in 'YYYY-MM-DD'.
    - timeframe (str): Timeframe for the bars.

    Returns:
    - df (pd.DataFrame or None): Cached DataFrame or None if not cached or corrupted.
    """
    cache_path = get_cache_path(symbol, start_date, end_date, timeframe)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            # Verify MultiIndex
            if isinstance(df.index, pd.MultiIndex) and df.index.names == ['symbol', 'timestamp']:
                logger.debug(f"Loaded cached data for {symbol} from {cache_path}.")
                return df
            else:
                logger.warning(f"Cached data for {symbol} does not have the correct MultiIndex. Deleting cache.")
                os.remove(cache_path)
                return None
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}. Deleting corrupted cache.")
            os.remove(cache_path)
            return None
    return None


def cache_bars(symbol, start_date, end_date, timeframe='1Day', df=None):
    """
    Caches fetched bars for a symbol.

    Parameters:
    - symbol (str): Stock symbol.
    - start_date (str): Start date in 'YYYY-MM-DD'.
    - end_date (str): End date in 'YYYY-MM-DD'.
    - timeframe (str): Timeframe for the bars.
    - df (pd.DataFrame): DataFrame to cache.
    """
    if df is not None:
        cache_path = get_cache_path(symbol, start_date, end_date, timeframe)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"Cached data for {symbol} to {cache_path}.")
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")


#####################################################################
# Data Fetching and Universe Selection
#####################################################################

def get_all_symbols(api):
    """
    Fetches all symbols (active and inactive) from Alpaca.

    Parameters:
    - api (tradeapi.REST): Alpaca API instance.

    Returns:
    - symbols (list): List of all symbols.
    """
    logger.debug("Fetching all symbols from Alpaca.")
    try:
        assets = api.list_assets(status=None)  # Fetch all assets regardless of status
        symbols = [asset.symbol for asset in assets if asset.tradable and asset.exchange in ['NYSE', 'NASDAQ']]
        logger.info(f"Fetched {len(symbols)} symbols from Alpaca.")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols from Alpaca: {e}")
        return []


def get_bars(api, symbols, start_date, end_date, timeframe='1Day'):
    """
    Fetches historical bars for the given symbols between start_date and end_date.
    Utilizes caching to store and retrieve fetched data.

    Parameters:
    - api (tradeapi.REST): Alpaca API instance.
    - symbols (list): List of stock symbols.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - timeframe (str): Timeframe for the bars.

    Returns:
    - data (pd.DataFrame): MultiIndex DataFrame with symbols and timestamps.
    """
    symbols = [str(s) for s in symbols if isinstance(s, str) or not pd.isnull(s)]
    logger.debug(f"Requesting bars for {len(symbols)} symbols from {start_date} to {end_date}.")
    if not symbols:
        logger.debug("No symbols provided to get_bars.")
        return pd.DataFrame()

    ensure_cache_dir()

    all_data = []
    chunk_size = 100
    symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    for idx, chunk in enumerate(symbol_chunks, start=1):
        if not chunk:
            continue

        logger.debug(f"Processing chunk {idx}/{len(symbol_chunks)}: {chunk}")
        chunk_data = []

        for sym in chunk:
            cached_df = load_cached_bars(sym, start_date, end_date, timeframe)
            if cached_df is not None:
                chunk_data.append(cached_df)
            else:
                logger.debug(f"Fetching data for {sym} from Alpaca.")
                try:
                    bars = api.get_bars(sym, timeframe, start=start_date, end=end_date, adjustment='split')
                    df = bars.df
                    if df is None or df.empty:
                        logger.debug(f"No data returned for {sym}.")
                        continue

                    # Normalize column names to lowercase
                    df.columns = [col.lower() for col in df.columns]
                    logger.debug(f"Normalized columns for {sym}: {df.columns.tolist()}")

                    # Verify presence of required columns
                    required_columns = {'close', 'volume'}
                    missing_columns = required_columns - set(df.columns)
                    if missing_columns:
                        logger.error(f"Missing columns for {sym}: {missing_columns}. Skipping this symbol.")
                        continue

                    # Ensure 'symbol' column exists
                    if 'symbol' not in df.columns:
                        df['symbol'] = sym
                        logger.debug(f"Added 'symbol' column for {sym}.")

                    # **Always reset the index to ensure 'timestamp' is a column**
                    df = df.reset_index()

                    # **Handle possible different timestamp column names**
                    if 'timestamp' not in df.columns:
                        if 'time' in df.columns:
                            df.rename(columns={'time': 'timestamp'}, inplace=True)
                            logger.debug(f"Renamed 'time' to 'timestamp' for {sym}.")
                        else:
                            logger.error(f"'timestamp' column missing for {sym}. Skipping.")
                            continue

                    # Set MultiIndex ['symbol', 'timestamp']
                    if 'symbol' in df.columns and 'timestamp' in df.columns:
                        df = df.set_index(['symbol', 'timestamp'])
                        logger.debug(f"Set MultiIndex ['symbol', 'timestamp'] for {sym}.")
                    else:
                        logger.error(f"Failed to set MultiIndex for {sym}. Missing 'symbol' or 'timestamp'. Skipping.")
                        continue

                    # Ensure timestamps are timezone-naive
                    if df.index.get_level_values('timestamp').tz is not None:
                        df.index = df.index.set_levels(
                            df.index.levels[1].tz_convert(None), level='timestamp'
                        )
                        logger.debug(f"Converted 'timestamp' to timezone-naive for {sym}.")

                    # Final check for MultiIndex
                    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['symbol', 'timestamp']:
                        logger.error(f"MultiIndex for {sym} is incorrect: {df.index.names}. Skipping.")
                        continue

                    # Cache the fetched data
                    cache_bars(sym, start_date, end_date, timeframe, df)

                    # Append to chunk_data
                    chunk_data.append(df)

                except Exception as e:
                    logger.error(f"Error fetching data for {sym}: {e}")
                    continue

        if chunk_data:
            try:
                concatenated_chunk = pd.concat(chunk_data)
                all_data.append(concatenated_chunk)
                logger.debug(f"Concatenated chunk {idx} with shape {concatenated_chunk.shape}.")
            except Exception as e:
                logger.error(f"Error concatenating chunk {idx}: {e}")
        else:
            logger.debug(f"No valid data fetched for chunk {idx}.")

    if not all_data:
        logger.debug("No bars data collected.")
        return pd.DataFrame()

    try:
        data = pd.concat(all_data)
    except Exception as e:
        logger.error(f"Error concatenating all_data: {e}")
        return pd.DataFrame()

    # Final verification of MultiIndex
    if not isinstance(data.index, pd.MultiIndex):
        logger.error("Aggregated data does not have a MultiIndex.")
        return pd.DataFrame()

    if data.index.names != ['symbol', 'timestamp']:
        logger.error(f"Aggregated MultiIndex levels are {data.index.names}, expected ['symbol', 'timestamp'].")
        return pd.DataFrame()

    # Extract 'close' and 'volume' prices
    try:
        close_prices = data['close'].unstack(level='symbol')
        volume_data = data['volume'].unstack(level='symbol')
    except KeyError as e:
        logger.error(f"Error unstacking 'symbol' level: {e}")
        return pd.DataFrame()

    logger.debug(f"Total data fetched: {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def get_all_price_data(api, all_symbols, start_date, end_date, timeframe='1Day'):
    """
    Fetches historical price data for all symbols from start_date to end_date.

    Parameters:
    - api (tradeapi.REST): Alpaca API instance.
    - all_symbols (list): List of all historical symbols.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - timeframe (str): Timeframe for the bars.

    Returns:
    - price_data (pd.DataFrame): MultiIndex DataFrame with symbols and timestamps.
    """
    logger.debug("Fetching all historical price data.")
    return get_bars(api, all_symbols, start_date, end_date, timeframe=timeframe)


def get_universe(historical_prices, current_date, universe_size=UNIVERSE_SIZE, min_avg_vol=MIN_AVG_VOL,
                 min_price=MIN_PRICE):
    """
    Selects the universe based on historical data up to the current_date.

    Parameters:
    - historical_prices (pd.DataFrame): All historical price data with MultiIndex ['symbol', 'timestamp'].
    - current_date (pd.Timestamp): The date up to which data is considered.
    - universe_size (int): Maximum number of symbols in the universe.
    - min_avg_vol (int): Minimum average volume required.
    - min_price (float): Minimum stock price required.

    Returns:
    - selected_symbols (list): List of selected stock symbols.
    """
    logger.debug(f"Selecting universe for date: {current_date}")

    # Ensure current_date is timezone-naive for comparison
    if current_date.tzinfo is not None:
        try:
            current_date = current_date.tz_localize(None)
            logger.debug("Converted 'current_date' to timezone-naive.")
        except TypeError:
            # If already timezone-naive, do nothing
            logger.debug("'current_date' is already timezone-naive.")

    # Filter data up to the current_date
    try:
        # If 'timestamp' is part of the index, filter accordingly
        if 'timestamp' in historical_prices.index.names:
            # Extract the 'timestamp' level
            unique_timestamps = historical_prices.index.levels[historical_prices.index.names.index('timestamp')]

            # Ensure timestamps are timezone-naive
            if isinstance(unique_timestamps[0], pd.Timestamp) and unique_timestamps[0].tzinfo is not None:
                unique_timestamps = unique_timestamps.tz_convert(None)
                historical_prices = historical_prices.copy()
                historical_prices.index = pd.MultiIndex.from_arrays(
                    [historical_prices.index.get_level_values('symbol'), unique_timestamps],
                    names=['symbol', 'timestamp'])
                logger.debug("Converted 'timestamp' to timezone-naive for MultiIndex.")

            # Create a boolean mask where timestamp <= current_date
            mask = historical_prices.index.get_level_values('timestamp') <= current_date
            data_up_to_date = historical_prices[mask]
        else:
            # If 'timestamp' is not part of the index, use loc
            data_up_to_date = historical_prices.loc[:current_date]
    except KeyError as e:
        logger.error(f"Error filtering data up to {current_date}: {e}")
        return []

    if data_up_to_date.empty:
        logger.warning("No price data available up to the current date.")
        return []

    # Reset index to make 'symbol' a column
    if 'symbol' not in data_up_to_date.columns and 'symbol' in data_up_to_date.index.names:
        data_up_to_date = data_up_to_date.reset_index()
        logger.debug("Reset index to make 'symbol' a column.")

    # Calculate average volume over the lookback period
    avg_vol = data_up_to_date.groupby('symbol')['volume'].mean()
    liquid_symbols = avg_vol[avg_vol > min_avg_vol].index.tolist()

    if not liquid_symbols:
        logger.warning("No liquid symbols found based on average volume criteria.")
        return []

    # Get the last available close price up to the current_date
    last_day_data = data_up_to_date.groupby('symbol').tail(1)
    if 'symbol' not in last_day_data.columns:
        logger.warning(f"'symbol' column missing after reset index. Skipping universe selection.")
        return []
    last_day_close = last_day_data.set_index('symbol')['close']
    viable = last_day_close[last_day_close > min_price].index.tolist()

    # Intersection of liquid symbols and viable symbols
    liquid_viable_symbols = list(set(liquid_symbols).intersection(set(viable)))

    if not liquid_viable_symbols:
        logger.warning("No symbols passed the volume and price filters.")
        return []

    # Sort by average volume and select top N
    avg_vol_filtered = avg_vol.loc[liquid_viable_symbols].sort_values(ascending=False)
    selected = avg_vol_filtered.head(universe_size).index.tolist()

    logger.debug(f"Selected {len(selected)} symbols for the universe based on volume and price criteria.")
    return selected


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


#####################################################################
# Rebalance Dates and Rebalance Portfolio
#####################################################################

def get_rebalance_dates(close_prices, freq=REBALANCE_FREQ):
    """
    Determines the rebalance dates based on the specified frequency.

    Parameters:
    - close_prices (pd.DataFrame): DataFrame containing close prices with a DateTime index.
    - freq (str): Rebalance frequency code.

    Returns:
    - rebal_dates (list of pd.Timestamp): List of dates to rebalance the portfolio.
    """
    # Map frequency codes
    freq_map = {
        'W': 'W-MON',  # Weekly on Monday
        'MS': 'M',      # Monthly Start
        'QS': 'Q',      # Quarterly Start
        'AS': 'A',      # Annually Start
        'YS': 'A'       # Another Annually Start alias
    }

    period_freq = freq_map.get(freq, 'M')

    # Ensure index is timezone-naive
    if close_prices.index.tz is not None:
        close_prices = close_prices.copy()
        close_prices.index = close_prices.index.tz_localize(None)
        logger.debug("Converted close_prices index to timezone-naive.")

    periods = close_prices.index.to_period(period_freq).unique()

    rebal_dates = []
    for p in periods:
        period_data = close_prices[close_prices.index.to_period(period_freq) == p]
        if not period_data.empty:
            # Append the first date of the period as the rebalance date
            rebal_dates.append(period_data.index[0])

    return rebal_dates


def rebalance_portfolio(current_positions, target_weights, current_prices, portfolio_value, entry_prices,
                        position_entry_dates, current_date):
    """
    Determines the trades needed to rebalance the portfolio to the target weights,
    considering stop-loss, take-profit, and minimum holding periods.

    Parameters:
    - current_positions (dict): Current positions with symbol as key and shares as value.
    - target_weights (dict): Target weights for each symbol.
    - current_prices (pd.Series): Current prices with symbol as index.
    - portfolio_value (float): Total portfolio value.
    - entry_prices (dict): Average entry price for each symbol.
    - position_entry_dates (dict): Entry date for each position.
    - current_date (pd.Timestamp): Current date.

    Returns:
    - trades (dict): Trades to execute with symbol as key and shares to buy/sell as value.
    """
    logger.debug("Rebalancing portfolio.")
    trades = {}

    sum_weights = sum(target_weights.values())
    if sum_weights > 1:
        factor = 1.0 / sum_weights
        target_weights = {k: v * factor for k, v in target_weights.items()}
    target_weights = {k: min(v, MAX_POSITION_SIZE) for k, v in target_weights.items()}

    # Check stop-loss and take-profit conditions
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


#####################################################################
# Backtesting Engine
#####################################################################

def extract_price_data(all_price_data):
    """
    Attempts to extract 'close' and 'volume' columns from all_price_data,
    handling different casing and column naming conventions.

    Parameters:
    - all_price_data (pd.DataFrame): Fetched price data.

    Returns:
    - close_prices (pd.DataFrame): DataFrame of close prices.
    - volume_data (pd.DataFrame): DataFrame of volume data.
    """
    # Attempt to find 'close' and 'volume' columns regardless of case
    lower_columns = {col.lower(): col for col in all_price_data.columns}

    close_col = lower_columns.get('close')
    volume_col = lower_columns.get('volume')

    if not close_col or not volume_col:
        logger.error("'close' or 'volume' column not found in price data.")
        # For debugging, log all available columns
        logger.debug(f"Available columns: {all_price_data.columns.tolist()}")
        return None, None

    # Check if 'symbol' is part of the index
    if 'symbol' not in all_price_data.index.names:
        logger.error("'symbol' level not found in MultiIndex. Cannot unstack.")
        return None, None

    try:
        close_prices = all_price_data[close_col].unstack(level='symbol')
        volume_data = all_price_data[volume_col].unstack(level='symbol')
    except KeyError as e:
        logger.error(f"Error unstacking 'symbol' level: {e}")
        return None, None

    return close_prices, volume_data


def verify_multiindex(all_price_data):
    """
    Verifies that all symbols in the MultiIndex have both 'symbol' and 'timestamp'.

    Parameters:
    - all_price_data (pd.DataFrame): Fetched price data.

    Returns:
    - valid_data (pd.DataFrame): DataFrame with correct MultiIndex.
    """
    if not isinstance(all_price_data.index, pd.MultiIndex):
        logger.error("Data does not have a MultiIndex.")
        logger.debug(f"Index type: {type(all_price_data.index)}")
        return pd.DataFrame()

    expected_levels = {'symbol', 'timestamp'}
    actual_levels = set(all_price_data.index.names)

    if not expected_levels.issubset(actual_levels):
        missing_levels = expected_levels - actual_levels
        logger.error(f"Missing MultiIndex levels: {missing_levels}")
        return pd.DataFrame()

    return all_price_data


def inspect_cached_data(symbol, start_date, end_date, timeframe='1Day'):
    """
    Loads and inspects cached data for a given symbol.

    Parameters:
    - symbol (str): Stock symbol.
    - start_date (str): Start date in 'YYYY-MM-DD'.
    - end_date (str): End date in 'YYYY-MM-DD'.
    - timeframe (str): Timeframe for the bars.
    """
    cache_path = get_cache_path(symbol, start_date, end_date, timeframe)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            print(f"--- Inspecting Cached Data for {symbol} ---")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Index Names: {df.index.names}")
            print(df.head())
            print("\n")
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
    else:
        print(f"No cache found for {symbol} at {cache_path}")


def backtest(api, start_date=START_DATE, end_date=END_DATE):
    logger.info(f"Starting backtest from {start_date} to {end_date}.")

    # Fetch all symbols from Alpaca
    all_symbols = get_all_symbols(api)
    if not all_symbols:
        logger.error("No symbols fetched from Alpaca. Exiting backtest.")
        return None, [], []

    # Fetch all historical price data with caching
    all_price_data = get_all_price_data(api, all_symbols, start_date, end_date)
    if all_price_data.empty:
        logger.warning("No price data fetched for any symbols.")
        return None, [], []

    # **Additional Logging:** Inspect the structure of all_price_data
    logger.debug(f"all_price_data index names: {all_price_data.index.names}")
    logger.debug(f"all_price_data columns: {all_price_data.columns.tolist()}")

    # Verify MultiIndex
    all_price_data = verify_multiindex(all_price_data)
    if all_price_data.empty:
        logger.error("MultiIndex verification failed.")
        return None, [], []

    # **Final MultiIndex Check:**
    if all_price_data.index.names != ['symbol', 'timestamp']:
        logger.error(f"Final MultiIndex levels are {all_price_data.index.names}, expected ['symbol', 'timestamp'].")
        return None, [], []

    # Extract 'close' and 'volume' prices
    try:
        close_prices, volume_data = extract_price_data(all_price_data)
    except Exception as e:
        logger.error(f"Failed to extract price data: {e}")
        return None, [], []

    if close_prices is None or volume_data is None:
        logger.warning("Close prices or volume data extraction returned None.")
        return None, [], []

    close_prices = close_prices.dropna(axis=1, how='all')
    volume_data = volume_data.dropna(axis=1, how='all')

    logger.info(f"Fetched close prices for {close_prices.shape[1]} symbols.")

    # Determine rebalance dates
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
    for rebalance_date in rebal_dates:
        logger.debug(f"Processing rebalance date: {rebalance_date}")

        # Select universe based on data up to the rebalance_date
        selected_universe = get_universe(
            historical_prices=all_price_data,
            current_date=rebalance_date,
            universe_size=UNIVERSE_SIZE,
            min_avg_vol=MIN_AVG_VOL,
            min_price=MIN_PRICE
        )

        if not selected_universe:
            logger.warning(f"No universe selected on {rebalance_date}. Skipping rebalance.")
            continue

        # Get price data up to the rebalance_date for momentum calculation
        try:
            price_window = close_prices.loc[:rebalance_date]
        except KeyError as e:
            logger.error(f"Error accessing price data up to {rebalance_date}: {e}")
            continue

        selected_prices = price_window[selected_universe]

        # Select stocks based on momentum
        selected = select_stocks(selected_prices, n=NUM_STOCKS, lookback=LOOKBACK_PERIOD)

        if not selected:
            logger.warning(f"No stocks selected based on momentum on {rebalance_date}.")
            continue

        # Market regime filter
        if MARKET_REGIME_SYMBOL in price_window.columns:
            spy_prices = price_window[MARKET_REGIME_SYMBOL].dropna()
            if len(spy_prices) > 200:
                spy_sma_200 = spy_prices.rolling(200).mean().iloc[-1]
                if spy_prices.iloc[-1] < spy_sma_200:
                    logger.debug("Market regime negative, halving target weights.")
                    selected = selected[:max(1, int(len(selected) / 2))]

        if len(selected) > 0:
            target_weights = volatility_scale(price_window, selected, lookback=LOOKBACK_PERIOD)
        else:
            target_weights = {}

        # Calculate portfolio value before rebalancing
        pos_value = sum([positions.get(s, 0) * close_prices.loc[rebalance_date, s]
                         for s in positions if s in close_prices.columns])
        total_equity = cash + pos_value

        # Rebalance portfolio
        trades = rebalance_portfolio(
            current_positions=positions,
            target_weights=target_weights,
            current_prices=close_prices.loc[rebalance_date],
            portfolio_value=total_equity,
            entry_prices=entry_prices,
            position_entry_dates=position_entry_dates,
            current_date=rebalance_date
        )

        for sym, shares_delta in trades.items():
            if sym not in close_prices.columns or pd.isna(close_prices.loc[rebalance_date, sym]):
                logger.debug(f"No price data for {sym} on {rebalance_date}, skipping trade.")
                continue

            # Execute trade
            trade_price = close_prices.loc[rebalance_date, sym]
            trade_price *= (1 + SLIPPAGE_BPS / 10000.0) if shares_delta > 0 else (1 - SLIPPAGE_BPS / 10000.0)
            trade_price = round(trade_price, 2)  # Round to 2 decimal places

            if shares_delta > 0:
                # Buy
                trade_cost = shares_delta * trade_price + COMMISSION_PER_TRADE
                if cash - trade_cost < 0:
                    max_shares = int((cash - COMMISSION_PER_TRADE) / trade_price)
                    if max_shares <= 0:
                        logger.debug(f"Not enough cash to buy {sym}. Skipping.")
                        continue
                    trade_cost = max_shares * trade_price + COMMISSION_PER_TRADE
                    shares_delta = max_shares

                cash -= trade_cost
                old_shares = positions.get(sym, 0)
                new_shares = old_shares + shares_delta
                positions[sym] = new_shares

                # Update average entry price
                if old_shares == 0:
                    entry_prices[sym] = trade_price
                    position_entry_dates[sym] = rebalance_date
                else:
                    entry_prices[sym] = ((entry_prices[sym] * old_shares) + (trade_price * shares_delta)) / new_shares

                # Record Execution Log
                execution_log.append({
                    'Date': rebalance_date,
                    'Symbol': sym,
                    'Action': 'Buy',
                    'Shares': shares_delta,
                    'Price': trade_price,
                    'Commission': COMMISSION_PER_TRADE,
                    'Total Cost/Proceeds': shares_delta * trade_price + COMMISSION_PER_TRADE,
                    'Cash After Trade': cash,
                    'Portfolio Value After Trade': cash + sum(
                        [positions.get(s, 0) * close_prices.loc[rebalance_date, s] for s in positions if s in close_prices.columns])
                })

            elif shares_delta < 0:
                # Sell
                shares_to_sell = abs(shares_delta)
                current_shares = positions.get(sym, 0)
                if shares_to_sell > current_shares:
                    shares_to_sell = current_shares

                trade_proceeds = shares_to_sell * trade_price - COMMISSION_PER_TRADE
                cash += trade_proceeds
                new_shares = current_shares - shares_to_sell
                positions[sym] = new_shares

                # Record Execution Log
                execution_log.append({
                    'Date': rebalance_date,
                    'Symbol': sym,
                    'Action': 'Sell',
                    'Shares': shares_to_sell,
                    'Price': trade_price,
                    'Commission': COMMISSION_PER_TRADE,
                    'Total Cost/Proceeds': shares_to_sell * trade_price - COMMISSION_PER_TRADE,
                    'Cash After Trade': cash,
                    'Portfolio Value After Trade': cash + sum(
                        [positions.get(s, 0) * close_prices.loc[rebalance_date, s] for s in positions if s in close_prices.columns])
                })

                # Record Trade Log for realized profit
                profit_per_share = trade_price - entry_prices.get(sym, 0)
                realized_profit = profit_per_share * shares_to_sell
                trade_log.append({
                    'Symbol': sym,
                    'Entry Date': position_entry_dates.get(sym, np.nan),
                    'Exit Date': rebalance_date,
                    'Entry Price': entry_prices.get(sym, 0),
                    'Exit Price': trade_price,
                    'Shares': shares_to_sell,
                    'Profit': realized_profit
                })

                if new_shares == 0:
                    del entry_prices[sym]
                    del position_entry_dates[sym]

            logger.debug(f"Executed trade for {sym}: {shares_delta} shares at {trade_price}")

        # Record portfolio history
        pos_value = sum([positions.get(s, 0) * close_prices.loc[rebalance_date, s]
                         for s in positions if s in close_prices.columns])
        total_equity = cash + pos_value
        portfolio_history.append([rebalance_date, total_equity, cash, pos_value])

    # After all rebalance dates, close any remaining positions
    logger.info("Closing any remaining open positions at the end of backtest period.")
    final_date = close_prices.index[-1]
    for sym, shares in list(positions.items()):
        if shares != 0 and sym in close_prices.columns:
            logger.debug(f"Closing position for {sym}: {shares} shares at {final_date}")
            exit_price = close_prices.loc[final_date, sym]
            trade_price = exit_price * (1 - SLIPPAGE_BPS / 10000.0)  # Apply slippage for selling
            trade_price = round(trade_price, 2)

            trade_proceeds = shares * trade_price - COMMISSION_PER_TRADE
            cash += trade_proceeds
            profit = (trade_price - entry_prices.get(sym, 0)) * shares
            trade_log.append({
                'Symbol': sym,
                'Entry Date': position_entry_dates.get(sym, np.nan),
                'Exit Date': final_date,
                'Entry Price': entry_prices.get(sym, 0),
                'Exit Price': trade_price,
                'Shares': shares,
                'Profit': profit
            })

            # Record Execution Log for closing the position
            execution_log.append({
                'Date': final_date,
                'Symbol': sym,
                'Action': 'Sell',
                'Shares': shares,
                'Price': trade_price,
                'Commission': COMMISSION_PER_TRADE,
                'Total Cost/Proceeds': shares * trade_price - COMMISSION_PER_TRADE,
                'Cash After Trade': cash,
                'Portfolio Value After Trade': cash
            })

            # Remove the position
            del positions[sym]
            del entry_prices[sym]
            del position_entry_dates[sym]

    # Create portfolio history DataFrame
    df_history = pd.DataFrame(portfolio_history, columns=['Date', 'Equity', 'Cash', 'Positions_Value'])
    df_history.set_index('Date', inplace=True)
    logger.info("Backtest complete.")

    return df_history, trade_log, execution_log


#####################################################################
# Performance Metrics
#####################################################################

def performance_metrics(df, trade_log, risk_free_rate=0.0):
    """
    Calculates various performance metrics for the backtest.

    Parameters:
    - df (pd.DataFrame): Portfolio history with 'Equity' column.
    - trade_log (list): List of trade dictionaries.
    - risk_free_rate (float): Annual risk-free rate for Sharpe and Sortino ratios.

    Returns:
    - metrics (dict): Dictionary of performance metrics.
    """
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
        logger.warning(
            f"Discrepancy detected: Realized Profit ({total_realized_profit}) != Portfolio Profit ({total_portfolio_profit})")
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

    # Optionally, inspect specific cached data for debugging
    # Uncomment the following lines to inspect a particular symbol
    # inspect_cached_data('CNSP', START_DATE, END_DATE)
    # inspect_cached_data('GNPX', START_DATE, END_DATE)
    # ... add more symbols as needed

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
