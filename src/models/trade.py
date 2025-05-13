from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class TradeType(str, Enum):
    short = 'short'
    long = 'long'

class TradeStatus(str, Enum):
    open = 'Open'
    closed_stop = 'Closed (Stop)'
    closed_eod = 'Closed (EOD)'

class Trade(BaseModel):
    day_date_str: str
    trade_num_day: int
    type: TradeType | None
    entry_time: datetime
    entry_price: float
    stop_price: float
    units: float
    position_value_usd: float
    status: TradeStatus
    ref_high_active: float 
    ref_low_active: float
    max_favorable_excursion_price: float
    min_adverse_excursion_price: float
    max_profit_before_close: float = 0.0
    pnl: float | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None

    class Config:
        from_attributes = True
