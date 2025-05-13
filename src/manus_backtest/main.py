"""
Trading Strategy Backtester

This script backtests a specific trading strategy based on daily reference candles,
entry signals, stop losses, and fixed risk position sizing.
"""
import pandas as pd
import numpy as np

from models.trade import Trade, TradeStatus, TradeType

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

def format_results_to_table(daily_results):
    """Formats the daily results into a pandas DataFrame as requested."""
    if not daily_results:
        return pd.DataFrame()

    table_data = []
    for day_res in daily_results:
        for i in range(len(day_res['Trades_Info'])):
            row = {
                'Day': day_res['Day'],
                'Reference_High': day_res['Ref_High'],
                'Reference_Low': day_res['Ref_Low'],
                'Trade_Number': i,
            }
            trade = day_res['Trades_Info'][i]
            row[f'Type'] = trade.type
            row[f'Entry_Price'] = trade.entry_price
            row[f'Stop_Price'] = trade.stop_price
            row[f'Pos_Size_USD'] = trade.position_value_usd or np.nan
            row[f"Pos_Size_Units"] = trade.units or np.nan
            row[f"Max_Profit_Before_Close"] = trade.max_profit_before_close or 0.0
            row[f"Result_EOD_PNL"] = trade.pnl or 0.0
            row[f"Status"] = trade.status.value or ""
            table_data.append(row)
    
    return pd.DataFrame(table_data)

def perform_analysis(all_trades_raw, risk_per_trade_fixed=RISK_AMOUNT_DOLLARS):
    """Performs additional analysis as requested by the user."""
    if not all_trades_raw:
        return "No trades to analyze."

    analysis_lines = []
    analysis_lines.append("--- Trade Analysis ---")

    # R-multiple analysis (3x, 5x, 10x risk)
    r_multiples_achieved = {3: 0, 5: 0, 10: 0}
    for trade in all_trades_raw:
        if trade.status not in [TradeStatus.closed_stop, TradeStatus.closed_eod]: # Should not happen if logic is correct
            continue
        
        potential_profit = 0
        if trade.type == TradeType.long:
            potential_profit = (trade.max_favorable_excursion_price - trade.entry_price) * trade.units
        elif trade.type == TradeType.short:
            potential_profit = (trade.entry_price - trade.max_favorable_excursion_price) * trade.units

        for r_val in r_multiples_achieved.keys():
            if potential_profit >= r_val * risk_per_trade_fixed:
                r_multiples_achieved[r_val] += 1
    
    analysis_lines.append("\nProfit Potential (R-multiples of $100 risk):")
    for r_val, count in r_multiples_achieved.items():
        analysis_lines.append(f"  Trades reaching at least {r_val}R ({r_val*risk_per_trade_fixed:.2f} profit): {count} ({(count/len(all_trades_raw)*100 if len(all_trades_raw)>0 else 0):.2f}% of trades)")

    # Day of the week analysis
    # trades_df = pd.DataFrame(all_trades_raw)
    trades_df = pd.DataFrame([{
        **trade.__dict__,
        "type": trade.type.value or "",
        "status": trade.status.value or "",
    } for trade in all_trades_raw])

    if not trades_df.empty and 'entry_time' in trades_df.columns:
        trades_df['entry_day_of_week'] = trades_df['entry_time'].dt.day_name()
        pnl_by_dow = trades_df.groupby('entry_day_of_week')['pnl'].sum().sort_values(ascending=False)
        analysis_lines.append("\nTotal P&L by Day of the Week (Entry Day):")
        analysis_lines.append(pnl_by_dow.to_string())
    
    # Trade direction analysis
    if not trades_df.empty:
        pnl_by_direction = trades_df.groupby('type')['pnl'].agg(['sum', 'mean', 'count'])
        analysis_lines.append("\nPerformance by Trade Direction:")
        analysis_lines.append(pnl_by_direction.to_string())

    # Trades per day analysis
    if not trades_df.empty and 'day_date_str' in trades_df.columns:
        trades_per_day_series = trades_df.groupby('day_date_str')['trade_num_day'].max()
        avg_trades_per_day = trades_per_day_series.mean()
        max_trades_in_a_day = trades_per_day_series.max()
        analysis_lines.append(f"\nAverage trades per trading day: {avg_trades_per_day:.2f}")
        analysis_lines.append(f"Maximum trades in a single day: {max_trades_in_a_day}")

    return "\n".join(analysis_lines)


if __name__ == "__main__":
    print(f"Loading data from {DATA_FILEPATH}...")
    df_ohlc = load_data(DATA_FILEPATH)

    if df_ohlc is not None and not df_ohlc.empty:
        print("Running backtest...")
        daily_summary, raw_trades = run_backtest(df_ohlc.copy()) # Use copy to avoid modifying original df
        
        if daily_summary:
            print("\n--- Daily Trade Summary Table ---")
            results_table_df = format_results_to_table(daily_summary)
            print(results_table_df.to_string())
            try:
                results_table_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"\nResults table saved to {OUTPUT_CSV_PATH}")
            except Exception as e:
                print(f"Error saving results table to CSV: {e}")
        else:
            print("No daily summary generated.")

        if raw_trades:
            analysis_string = perform_analysis(raw_trades)
            print("\n" + analysis_string)
            try:
                # --- New Analysis: Optimal Max Trades Per Day ---
                analysis_lines = []
                analysis_lines.append("\n--- Optimal Max Trades Per Day Analysis ---")
                if raw_trades:
                    trades_df_for_cap_analysis = pd.DataFrame(raw_trades)
                    if not trades_df_for_cap_analysis.empty and 'pnl' in trades_df_for_cap_analysis.columns and 'day_date_str' in trades_df_for_cap_analysis.columns and 'entry_time' in trades_df_for_cap_analysis.columns:
                        trades_df_for_cap_analysis['entry_time'] = pd.to_datetime(trades_df_for_cap_analysis['entry_time'])
                        trades_df_for_cap_analysis = trades_df_for_cap_analysis.sort_values(by=['day_date_str', 'entry_time'])
                        trades_df_for_cap_analysis['trade_num_in_day'] = trades_df_for_cap_analysis.groupby('day_date_str').cumcount() + 1
                        
                        avg_pnl_by_nth_trade = trades_df_for_cap_analysis.groupby('trade_num_in_day')['pnl'].agg(['mean', 'sum', 'count'])
                        analysis_lines.append("\nAverage P&L by Nth trade of the day (across all days):")
                        for index, row_nth_trade in avg_pnl_by_nth_trade.iterrows():
                            analysis_lines.append(f"  Trade #{index}: Avg P&L {row_nth_trade.mean:.2f} (Total P&L: {row_nth_trade.sum:.2f}, Count: {row_nth_trade.count})")

                        analysis_lines.append("\nCumulative P&L if stopping after N trades per day:")
                        max_trades_observed_in_any_day = trades_df_for_cap_analysis['trade_num_in_day'].max() if not trades_df_for_cap_analysis.empty else 0
                        baseline_total_pnl = trades_df_for_cap_analysis['pnl'].sum()
                        analysis_lines.append(f"  Baseline (all trades taken): Total P&L {baseline_total_pnl:.2f}")

                        if max_trades_observed_in_any_day > 0:
                            for n_cap in range(1, int(max_trades_observed_in_any_day) + 1):
                                capped_pnl_total = 0
                                for day_str_loop, daily_trades_group in trades_df_for_cap_analysis.groupby('day_date_str'):
                                    capped_pnl_total += daily_trades_group.head(n_cap)['pnl'].sum()
                                analysis_lines.append(f"  Capping at {n_cap} trade(s)/day: Total P&L {capped_pnl_total:.2f}")
                    else:
                        analysis_lines.append("  DataFrame empty or required columns ('pnl', 'day_date_str', 'entry_time') missing for optimal max trades per day analysis.")
                else:
                    analysis_lines.append("  No trades available for optimal max trades per day analysis.")
                # --- End of New Optimal Max Trades Per Day Analysis ---

                # --- New Analysis: Max Profit Before Close for Take-Profit Analysis ---
                analysis_lines.append("\n--- Max Profit Before Close for Take-Profit Analysis ---")
                if raw_trades:
                    trades_df_for_tp_analysis = pd.DataFrame(raw_trades)
                    if not trades_df_for_tp_analysis.empty and 'max_profit_before_close' in trades_df_for_tp_analysis.columns and pd.api.types.is_numeric_dtype(trades_df_for_tp_analysis['max_profit_before_close']):
                        trades_df_for_tp_analysis['max_profit_R'] = trades_df_for_tp_analysis['max_profit_before_close'] / RISK_AMOUNT_DOLLARS
                        analysis_lines.append("Distribution of Max Profit Before Close (in R-multiples):")
                        analysis_lines.append(trades_df_for_tp_analysis['max_profit_R'].describe().to_string())

                        r_levels_to_check = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
                        analysis_lines.append("\nPercentage of trades where Max Profit Before Close reached at least X*R:")
                        total_trades_count = len(trades_df_for_tp_analysis)
                        if total_trades_count > 0:
                            for r_level in r_levels_to_check:
                                count_reaching_r = trades_df_for_tp_analysis[trades_df_for_tp_analysis['max_profit_R'] >= r_level].shape[0]
                                percentage_reaching_r = (count_reaching_r / total_trades_count) * 100
                                analysis_lines.append(f"  {r_level:.1f}R: {percentage_reaching_r:.2f}% of trades ({count_reaching_r} trades)")
                        else:
                            analysis_lines.append("  No trades to analyze for R-multiple distribution.")
                    else:
                        analysis_lines.append("  'max_profit_before_close' column not found, not numeric, or DataFrame empty for take-profit analysis.")
                else:
                    analysis_lines.append("  No trades available for take-profit analysis.")
                # --- End of Max Profit Before Close Analysis ---

                with open(ANALYSIS_OUTPUT_PATH, "w") as f:
                    f.write(analysis_string)
                print(f"Analysis saved to {ANALYSIS_OUTPUT_PATH}")
            except Exception as e:
                print(f"Error saving analysis to file: {e}")
            else:
                print("No trades were made to perform analysis.")
        else:
            print("Failed to load data or data is empty. Backtest aborted.")

