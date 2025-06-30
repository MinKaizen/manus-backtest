from abc import ABC, abstractmethod
from models.ohlc_candle import OHLCCandle
from pandas import DataFrame
from typing import TypeVar, Generic

T = TypeVar("T")

class Indicator(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, df: DataFrame):
        pass

    @abstractmethod
    def value(self, timestamp: float) -> T:
        pass

