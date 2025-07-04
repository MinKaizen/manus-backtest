"""
Report generation module for trading strategy backtester.

This module handles all report generation functionality including:
- Formatting results into tables
- Performing trade analysis
- Generating output files
"""
import pandas as pd
import numpy as np
import os
from typing import List

from models.trade import Trade, TradeStatus, TradeType
from models.daily_result import DailyResult


class BacktestReporter:
    """Handles all report generation for backtest results."""
    
    def __init__(self, risk_amount_dollars: float = 100.0):
        self.risk_amount_dollars = risk_amount_dollars
    
    def format_results_to_table(self, daily_results: List[DailyResult]) -> pd.DataFrame:
        """Formats the daily results into a pandas DataFrame as requested."""
        if not daily_results:
            return pd.DataFrame()

        table_data = []
        for day_res in daily_results:
            for i in range(len(day_res.Trades_Info)):
                row = {
                    'Day': day_res.Day,
                    'Reference_High': day_res.Ref_High,
                    'Reference_Low': day_res.Ref_Low,
                    'Trade_Number': i,
                }
                trade = day_res.Trades_Info[i]
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

    def perform_analysis(self, all_trades_raw: List[Trade]) -> str:
        """Performs additional analysis as requested by the user."""
        if not all_trades_raw:
            return "No trades to analyze."

        analysis_lines = []
        analysis_lines.append("--- Trade Analysis ---")

        # R-multiple analysis (3x, 5x, 10x risk)
        r_multiples_achieved = {3: 0, 5: 0, 10: 0}
        for trade in all_trades_raw:
            if trade.status not in [TradeStatus.closed_stop, TradeStatus.closed_eod]:
                continue
            
            potential_profit = 0
            if trade.type == TradeType.long:
                potential_profit = (trade.max_favorable_excursion_price - trade.entry_price) * trade.units
            elif trade.type == TradeType.short:
                potential_profit = (trade.entry_price - trade.max_favorable_excursion_price) * trade.units

            for r_val in r_multiples_achieved.keys():
                if potential_profit >= r_val * self.risk_amount_dollars:
                    r_multiples_achieved[r_val] += 1
        
        analysis_lines.append("\nProfit Potential (R-multiples of $100 risk):")
        for r_val, count in r_multiples_achieved.items():
            analysis_lines.append(f"  Trades reaching at least {r_val}R ({r_val*self.risk_amount_dollars:.2f} profit): {count} ({(count/len(all_trades_raw)*100 if len(all_trades_raw)>0 else 0):.2f}% of trades)")

        # Day of the week analysis
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

    def generate_extended_analysis(self, all_trades_raw: List[Trade]) -> str:
        """Generates extended analysis including optimal max trades and take-profit analysis."""
        analysis_lines = []
        
        # --- Optimal Max Trades Per Day Analysis ---
        analysis_lines.append("\n--- Optimal Max Trades Per Day Analysis ---")
        if all_trades_raw:
            trades_df_for_cap_analysis = pd.DataFrame(all_trades_raw)
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

        # --- Max Profit Before Close for Take-Profit Analysis ---
        analysis_lines.append("\n--- Max Profit Before Close for Take-Profit Analysis ---")
        if all_trades_raw:
            trades_df_for_tp_analysis = pd.DataFrame(all_trades_raw)
            if not trades_df_for_tp_analysis.empty and 'max_profit_before_close' in trades_df_for_tp_analysis.columns and pd.api.types.is_numeric_dtype(trades_df_for_tp_analysis['max_profit_before_close']):
                trades_df_for_tp_analysis['max_profit_R'] = trades_df_for_tp_analysis['max_profit_before_close'] / self.risk_amount_dollars
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

        return "\n".join(analysis_lines)

    def save_reports(self, daily_results: List[DailyResult], all_trades_raw: List[Trade], 
                    csv_output_path: str, analysis_output_path: str) -> None:
        """Saves both CSV and analysis reports to files."""
        # Create output directories if they don't exist
        csv_dir = os.path.dirname(csv_output_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        
        analysis_dir = os.path.dirname(analysis_output_path)
        if analysis_dir and not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir, exist_ok=True)
        
        # Save CSV report
        if daily_results:
            results_table_df = self.format_results_to_table(daily_results)
            try:
                results_table_df.to_csv(csv_output_path, index=False)
                print(f"Results table saved to {csv_output_path}")
            except Exception as e:
                print(f"Error saving results table to CSV: {e}")
        
        # Save analysis report
        if all_trades_raw:
            try:
                analysis_string = self.perform_analysis(all_trades_raw)
                
                with open(analysis_output_path, "w") as f:
                    f.write(analysis_string)
                print(f"Analysis saved to {analysis_output_path}")
            except Exception as e:
                print(f"Error saving analysis to file: {e}")

    def print_reports(self, daily_results: List[DailyResult], all_trades_raw: List[Trade]) -> None:
        """Prints reports to console."""
        if daily_results:
            print("\n--- Daily Trade Summary Table ---")
            results_table_df = self.format_results_to_table(daily_results)
            print(results_table_df.to_string())
        else:
            print("No daily summary generated.")

        if all_trades_raw:
            analysis_string = self.perform_analysis(all_trades_raw)
            extended_analysis = self.generate_extended_analysis(all_trades_raw)
            print("\n" + analysis_string)
            print(extended_analysis)
        else:
            print("No trades were made to perform analysis.")