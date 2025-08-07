"""
Trading Strategy Backtester

This script backtests a specific trading strategy based on daily reference candles,
entry signals, stop losses, and fixed risk position sizing.
"""
import pandas as pd
import os
import glob
from pathlib import Path

from indicators.BCERefs import BCERefsIndicator
from models.trade import Trade, TradeStatus, TradeType, create_trade
from models.daily_result import DailyResult
from models.backtest_config import BacktestConfig
from models.ohlc_candle import OHLCCandle
from manus_backtest.reports import BacktestReporter

config = BacktestConfig(
    risk_amount_dollars=100.0,
    output_csv_path="./output/trade_results.csv",
    analysis_output_path="./output/trade_analysis.txt",
    data_filepath="./BTCUSDT_OHLC.csv"
)

def load_data(filepath):
    """Loads OHLC data from CSV and prepares it."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None

    # Convert all column names to lowercase for consistent access
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    datetime_col_to_set_as_index = None # Will store the name of the column successfully converted to datetime

    # Try to identify and parse the timestamp column
    # Common names: 'time', 'unix', 'timestamp'
    # Order of preference: 'time' (if numeric UNIX), 'unix', 'timestamp' (numeric or string)

    if 'time' in df.columns and pd.api.types.is_numeric_dtype(df['time']):
        # Assuming 'time' column contains UNIX timestamps in seconds
        try:
            df['datetime_col_temp'] = pd.to_datetime(df['time'], unit='s', utc=True)
            datetime_col_to_set_as_index = 'datetime_col_temp'
        except Exception as e:
            print(f"Warning: 'time' column is numeric but failed conversion to datetime: {e}")
            pass # Continue to check other columns
            
    if not datetime_col_to_set_as_index and 'unix' in df.columns and pd.api.types.is_numeric_dtype(df['unix']):
        try:
            df['datetime_col_temp'] = pd.to_datetime(df['unix'], unit='s', utc=True)
            datetime_col_to_set_as_index = 'datetime_col_temp'
        except Exception as e:
            print(f"Warning: 'unix' column is numeric but failed conversion to datetime: {e}")
            pass

    if not datetime_col_to_set_as_index and 'timestamp' in df.columns:
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            # Check if it's seconds or milliseconds
            try:
                # Heuristic: if the number is very large, it's likely milliseconds, else seconds.
                # This might need adjustment based on typical timestamp ranges.
                if df['timestamp'].iloc[0] > 1e11:  # Arbitrary threshold for ms vs s
                    df['datetime_col_temp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                else:
                    df['datetime_col_temp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                datetime_col_to_set_as_index = 'datetime_col_temp'
            except Exception as e:
                print(f"Warning: 'timestamp' column is numeric but failed conversion to datetime: {e}")
                pass
        else:
            # Try to parse as a datetime string
            try:
                df['datetime_col_temp'] = pd.to_datetime(df['timestamp'], utc=True)
                datetime_col_to_set_as_index = 'datetime_col_temp'
            except Exception as e:
                print(f"Warning: 'timestamp' column is string but failed conversion to datetime: {e}")
                pass
    
    # Fallback for 'time' column if it wasn't numeric but might be a datetime string
    if not datetime_col_to_set_as_index and 'time' in df.columns:
        try:
            df['datetime_col_temp'] = pd.to_datetime(df['time'], utc=True)
            datetime_col_to_set_as_index = 'datetime_col_temp'
        except Exception as e:
            print(f"Warning: 'time' column (as string) failed conversion to datetime: {e}")
            pass
        
    if datetime_col_to_set_as_index:
        df.set_index(datetime_col_to_set_as_index, inplace=True)
        df.index.name = 'timestamp'  # Standardize index name for the rest of the script
        # If we created a temporary column and it's different from a potential original 'timestamp' column that was not used for index
        if datetime_col_to_set_as_index != 'timestamp' and datetime_col_to_set_as_index in df.columns:
             df.drop(columns=[datetime_col_to_set_as_index], inplace=True, errors='ignore')

        # Ensure data is sorted by time
        df.sort_index(inplace=True)
        return df
    else:
        print("Error: Suitable timestamp column ('time', 'unix', or 'timestamp') not found or could not be parsed.")
        return None

def calculate_position_size(entry_price: float, stop_price: float, risk_amount_dollars: float) -> tuple[float, float]:
    """Calculates position size in units and dollar value."""
    if entry_price is None or stop_price is None:
        return 0, 0
    
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0: # Avoid division by zero
        return 0, 0

    position_size_units = risk_amount_dollars / risk_per_unit
    position_value_dollars = position_size_units * entry_price
    return position_size_units, position_value_dollars

def run_backtest(df: pd.DataFrame) -> tuple[list[DailyResult], list[Trade]]:
    """Runs the backtesting simulation."""
    if df is None or df.empty:
        print("Dataframe is empty, cannot run backtest.")
        return [], []

    all_trades_raw = [] # For detailed analysis
    daily_results_for_table = []

    bcerefs = BCERefsIndicator(df)

    # Group data by day (UTC midnight to midnight)
    # Resample to ensure we get all days, then group by date part
    # This handles days with no trades as well.
    # The strategy is day-based, so we process day by day.
    
    # Group data by day (UTC midnight to midnight)
    # Iterate directly over the resampled object which yields (timestamp, group_df) pairs
    for day_date, day_data in df.resample('D'):
        if day_data.empty:
            continue

        day_str = day_date.strftime('%Y-%m-%d')
        current_day_trades_info = []
        active_trade = None
        trade_counter_this_day = 0

        for candle_timestamp, current_candle in day_data.iterrows():
            ref_high, ref_low = bcerefs.value(candle_timestamp.timestamp())

            # 1. Manage existing trade (check for stop loss)
            if active_trade:
                stop_hit = False
                exit_price_sl = None
                current_pnl = 0

                if active_trade.type == TradeType.long:
                    if current_candle.low <= active_trade.stop_price:
                        stop_hit = True
                        exit_price_sl = active_trade.stop_price
                        current_pnl = (exit_price_sl - active_trade.entry_price) * active_trade.units
                elif active_trade.type == TradeType.short:
                    if current_candle.high >= active_trade.stop_price:
                        stop_hit = True
                        exit_price_sl = active_trade.stop_price
                        current_pnl = (active_trade.entry_price - exit_price_sl) * active_trade.units

                if stop_hit:
                    active_trade.exit_price = exit_price_sl
                    active_trade.pnl = current_pnl
                    active_trade.status = TradeStatus.closed_stop
                    active_trade.exit_time = candle_timestamp
                    current_day_trades_info.append(active_trade.model_copy())
                    all_trades_raw.append(active_trade.model_copy())
                    active_trade = None

            # 2. Check for new entry signals if no active trade
            if not active_trade:
                is_buy_signal = current_candle.close > ref_high
                is_sell_signal = current_candle.close < ref_low
                entry_price = None
                stop_price_new_trade = None
                trade_type = None

                if is_buy_signal:
                    entry_price = current_candle.close
                    stop_price_new_trade = ref_low
                    trade_type = TradeType.long
                    if entry_price <= stop_price_new_trade: # Invalid trade condition
                        entry_price = None 
                elif is_sell_signal:
                    entry_price = current_candle.close
                    stop_price_new_trade = ref_high
                    trade_type = TradeType.short
                    if entry_price >= stop_price_new_trade: # Invalid trade condition
                        entry_price = None
                
                if entry_price and stop_price_new_trade and trade_type != None:
                    trade_counter_this_day += 1
                    units, value_dollars = calculate_position_size(entry_price, stop_price_new_trade, config.risk_amount_dollars)
                    
                    if units > 0: # Valid position
                        active_trade = create_trade(
                            day_date_str=day_str,
                            trade_num_day=trade_counter_this_day,
                            trade_type=trade_type,
                            entry_time=candle_timestamp,
                            entry_price=entry_price,
                            stop_price=stop_price_new_trade,
                            units=units,
                            position_value_usd=value_dollars,
                            ref_high_active=ref_high,
                            ref_low_active=ref_low
                        )
                   # Update MFE/MAE and Max Profit Before Close if trade is active
            if active_trade and active_trade.status == TradeStatus.open:
                current_potential_profit = 0
                if active_trade.type == TradeType.long:
                    active_trade.max_favorable_excursion_price = max(active_trade.max_favorable_excursion_price, current_candle.high)
                    active_trade.min_adverse_excursion_price = min(active_trade.min_adverse_excursion_price, current_candle.low)
                    current_potential_profit = (current_candle.high - active_trade.entry_price) * active_trade.units
                elif active_trade.type == TradeType.short:
                    active_trade.max_favorable_excursion_price = min(active_trade.max_favorable_excursion_price, current_candle.low) # Lower price is favorable for short
                    active_trade.min_adverse_excursion_price = max(active_trade.min_adverse_excursion_price, current_candle.high)
                    current_potential_profit = (active_trade.entry_price - current_candle.low) * active_trade.units
                active_trade.max_profit_before_close = max((active_trade.max_profit_before_close or 0.0), current_potential_profit)
        # At the end of the day, if there is an active trade, close it
        if active_trade and active_trade.status == TradeStatus.open: # ADDED THIS CHECK
            eod_close_price = day_data.iloc[-1].close
            pnl_eod = 0
            if active_trade.type == TradeType.long:
                pnl_eod = (eod_close_price - active_trade.entry_price) * active_trade.units
            elif active_trade.type == TradeType.short:
                pnl_eod = (active_trade.entry_price - eod_close_price) * active_trade.units
            
            active_trade.exit_price = eod_close_price
            active_trade.pnl = pnl_eod
            active_trade.status = TradeStatus.closed_eod
            active_trade.exit_time = day_data.index[-1]
            current_day_trades_info.append(active_trade.model_copy())
            all_trades_raw.append(active_trade.model_copy())
            active_trade = None # Reset active trade after closing EOD

        first_candle_timestamp = day_data.index[0].timestamp()
        ref_high, ref_low = bcerefs.value(first_candle_timestamp)
        daily_result = DailyResult(
            Day=day_str,
            Ref_High=ref_high,
            Ref_Low=ref_low,
            Trades_Info=current_day_trades_info
        )
        daily_results_for_table.append(daily_result)

    return daily_results_for_table, all_trades_raw

def process_single_file(filepath: str, output_dir: str = "./output") -> bool:
    """Process a single CSV file and generate reports."""
    filename = Path(filepath).stem
    print(f"Loading data from {filepath}...")
    df_ohlc = load_data(filepath)

    if df_ohlc is not None and not df_ohlc.empty:
        print(f"Running backtest for {filename}...")
        daily_summary, raw_trades = run_backtest(df_ohlc.copy())
        
        # Initialize reporter
        reporter = BacktestReporter(risk_amount_dollars=config.risk_amount_dollars)
        
        # Generate output filenames based on input filename
        output_csv_path = f"{output_dir}/trade_results_{filename}.csv"
        analysis_output_path = f"{output_dir}/trade_analysis_{filename}.txt"
        
        # Save reports to files
        reporter.save_reports(daily_summary, raw_trades, output_csv_path, analysis_output_path)
        print(f"Reports saved: {output_csv_path}, {analysis_output_path}")
        return True
    else:
        print(f"Failed to load data from {filepath} or data is empty. Skipping.")
        return False

if __name__ == "__main__":
    # Get the project root directory (two levels up from this file)
    current_dir = Path.cwd()
    if current_dir.name == "src":
        project_root = current_dir.parent
    elif (current_dir / "src").exists():
        project_root = current_dir
    else:
        project_root = Path(__file__).parent.parent.parent
    
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in input directory
    csv_files = glob.glob(str(input_dir / "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} directory.")
        print("Please place your CSV files in the input folder and try again.")
    else:
        print(f"Found {len(csv_files)} CSV file(s) in {input_dir}:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        print("\nProcessing files...")
        successful_runs = 0
        
        for csv_file in csv_files:
            if process_single_file(csv_file, str(output_dir)):
                successful_runs += 1
            print("-" * 50)
        
        print(f"\nCompleted processing {successful_runs}/{len(csv_files)} files successfully.")

