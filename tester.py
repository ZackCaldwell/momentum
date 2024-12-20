# test_script.py

import itertools
import pandas as pd
from updated import run_backtest_with_config, DEFAULT_CONFIG

def generate_parameter_grid(param_grid: dict):
    """
    Generates a list of configuration dictionaries based on the parameter grid.
    """
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        config = DEFAULT_CONFIG.copy()
        config.update(dict(zip(keys, v)))
        yield config

def main():
    # Define the parameters you want to test and their possible values
    parameter_grid = {
        "LOOKBACK_PERIOD": [30, 42, 60],
        "REBALANCE_FREQ": ['M', 'Q'],
        "NUM_STOCKS": [5, 10, 20],
        "UNIVERSE_SIZE": [100, 200, 300],
        "MIN_AVG_VOL": [1_000_000, 5_000_000, 10_000_000],
        "MIN_PRICE": [10.0, 20.0, 50.0],
        "COMMISSION_PER_TRADE": [0.5, 1.0, 2.0],
        "SLIPPAGE_BPS": [5, 10, 20],
        "STOP_LOSS_FACTOR": [0.75, 0.8, 0.85],
        "TAKE_PROFIT_FACTOR": [1.5, 2.0, 2.5],
        "MIN_HOLD_DAYS": [20, 30, 40],
    }

    # Generate all combinations (this can be large; consider limiting)
    configs = generate_parameter_grid(parameter_grid)

    # To store the results
    results = []

    # Optional: Limit the number of combinations for testing purposes
    max_combinations = 50  # Set to None to run all
    for idx, config in enumerate(configs, start=1):
        if max_combinations and idx > max_combinations:
            break
        print(f"Running backtest {idx} with configuration:")
        test_config = {k: config[k] for k in parameter_grid.keys()}
        print(test_config)
        metrics = run_backtest_with_config(config)
        if metrics:
            results.append(metrics)
        print(f"Completed backtest {idx}.\n")

    if not results:
        print("No backtest results were generated.")
        return

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Expand the Configuration column into separate columns
    config_df = df_results['Configuration'].apply(pd.Series)
    metrics_df = df_results.drop(columns=['Configuration'])

    final_df = pd.concat([config_df, metrics_df], axis=1)

    # Save to CSV for further analysis
    final_df.to_csv("backtest_results.csv", index=False)
    print("All backtests completed. Results saved to 'backtest_results.csv'.")

    # Optionally, analyze the best configuration based on a metric, e.g., Sharpe ratio
    if not final_df.empty:
        best_sharpe_idx = final_df['Sharpe'].idxmax()
        best_sharpe = final_df.loc[best_sharpe_idx]
        print("\nBest Configuration Based on Sharpe Ratio:")
        for k in parameter_grid.keys():
            print(f"{k}: {best_sharpe[k]}")
        print(f"Sharpe Ratio: {best_sharpe['Sharpe']:.2f}")
    else:
        print("No results to analyze.")

if __name__ == "__main__":
    main()
