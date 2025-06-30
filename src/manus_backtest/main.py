"""
Trading Strategy Backtester

This script backtests a specific trading strategy based on daily reference candles,
entry signals, stop losses, and fixed risk position sizing.
"""
import pandas as pd
import numpy as np

from models.trade import Trade, TradeStatus, TradeType
from manus_backtest.reports import BacktestReporter

RISK_AMOUNT_DOLLARS = 100.0
OUTPUT_CSV_PATH = "./output/trade_results.csv"
ANALYSIS_OUTPUT_PATH = "./output/trade_analysis.txt"
DATA_FILEPATH = "./BTCUSDT_OHLC.csv"

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

def calculate_reference_levels(first_candle):
    """Calculates reference high and low based on the first candle of the day."""
    o, h, l, c = first_candle.open, first_candle.high, first_candle.low, first_candle.close

    ref_high, ref_low = None, None

    # Rule: if the distance between open/close is less than 0.05%
    # Assuming 0.05% of the open price. Add check for open > 0.
    use_hl_for_ref = False
    if o > 0:
        if abs(o - c) / o < 0.0005:
            use_hl_for_ref = True
    else: # If open is 0, perhaps default to using H/L
        use_hl_for_ref = True 

    if use_hl_for_ref:
        ref_high = h
        ref_low = l
    else:
        if c > o:  # Up candle
            ref_high = c
            ref_low = o
        else:  # Down candle or equal
            ref_high = o
            ref_low = c
    return ref_high, ref_low

def calculate_position_size(entry_price, stop_price, risk_amount_dollars):
    """Calculates position size in units and dollar value."""
    if entry_price is None or stop_price is None:
        return 0, 0
    
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0: # Avoid division by zero
        return 0, 0

    position_size_units = risk_amount_dollars / risk_per_unit
    position_value_dollars = position_size_units * entry_price
    return position_size_units, position_value_dollars

def run_backtest(df):
    """Runs the backtesting simulation."""
    if df is None or df.empty:
        print("Dataframe is empty, cannot run backtest.")
        return [], []

    all_trades_raw = [] # For detailed analysis
    daily_results_for_table = []

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

        first_candle = day_data.iloc[0]
        ref_high, ref_low = calculate_reference_levels(first_candle)

        for candle_timestamp, current_candle in day_data.iterrows():
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
                    units, value_dollars = calculate_position_size(entry_price, stop_price_new_trade, RISK_AMOUNT_DOLLARS)
                    
                    if units > 0: # Valid position
                        active_trade = Trade(
                            day_date_str = day_str,
                            trade_num_day = trade_counter_this_day,
                            type = trade_type,
                            entry_time = candle_timestamp,
                            entry_price = entry_price,
                            stop_price = stop_price_new_trade,
                            units = units,
                            position_value_usd = value_dollars,
                            status = TradeStatus.open,
                            ref_high_active = ref_high, 
                            ref_low_active = ref_low,
                            max_favorable_excursion_price = entry_price, # For R-multiple analysis
                            min_adverse_excursion_price = entry_price,   # For R-multiple analysis
                            max_profit_before_close = 0.0 # Initialize max profit
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

        daily_results_for_table.append({
            'Day': day_str,
            'Ref_High': ref_high,
            'Ref_Low': ref_low,
            'Trades_Info': current_day_trades_info
        })

    return daily_results_for_table, all_trades_raw






if __name__ == "__main__":
    print(f"Loading data from {DATA_FILEPATH}...")
    df_ohlc = load_data(DATA_FILEPATH)

    if df_ohlc is not None and not df_ohlc.empty:
        print("Running backtest...")
        daily_summary, raw_trades = run_backtest(df_ohlc.copy())
        
        # Initialize reporter
        reporter = BacktestReporter(risk_amount_dollars=RISK_AMOUNT_DOLLARS)
        
        # Print reports to console
        reporter.print_reports(daily_summary, raw_trades)
        
        # Save reports to files
        reporter.save_reports(daily_summary, raw_trades, OUTPUT_CSV_PATH, ANALYSIS_OUTPUT_PATH)
    else:
        print("Failed to load data or data is empty. Backtest aborted.")

