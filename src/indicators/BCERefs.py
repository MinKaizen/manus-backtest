from pandas import DataFrame
from indicators.Indicator import Indicator

from models.ohlc_candle import OHLCCandle

class BCERefsIndicator(Indicator[tuple[float, float]]):
    _values: dict

    def __init__(self, df: DataFrame):
        current_day = None
        value = None
        self._values = {}

        for pd_timestamp, candle_data in df.iterrows():
            timestamp = pd_timestamp.timestamp()
            this_day = pd_timestamp.strftime('%Y-%m-%d')
            if current_day != this_day:
                candle = OHLCCandle(
                    open=candle_data.open,
                    high=candle_data.high,
                    low=candle_data.low,
                    close=candle_data.close,
                )

                value = self.calculate_reference_levels(candle)
                current_day = this_day
            self._values[timestamp] = value

    def value(self, timestamp: float) -> tuple[float, float]:
        if timestamp in self._values:
            return self._values[timestamp]
        raise KeyError(f"Timestamp '{timestamp} was not loaded into indicator BCERefs")

    def calculate_reference_levels(self, first_candle: OHLCCandle) -> tuple[float, float]:
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




