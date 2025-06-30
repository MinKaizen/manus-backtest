from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, validator

class TradeType(str, Enum):
    short = 'short'
    long = 'long'

class TradeStatus(str, Enum):
    open = 'Open'
    closed_stop = 'Closed (Stop)'
    closed_eod = 'Closed (EOD)'

class Trade(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    day_date_str: str = Field(..., description="Date string in YYYY-MM-DD format")
    trade_num_day: int = Field(..., ge=1, description="Trade number within the day")
    type: Optional[TradeType] = Field(None, description="Trade direction (long/short)")
    entry_time: datetime = Field(..., description="Entry timestamp")
    entry_price: float = Field(..., gt=0, description="Entry price")
    stop_price: float = Field(..., gt=0, description="Stop loss price")
    units: float = Field(..., gt=0, description="Position size in units")
    position_value_usd: float = Field(..., gt=0, description="Position value in USD")
    status: TradeStatus = Field(..., description="Current trade status")
    ref_high_active: float = Field(..., gt=0, description="Reference high for the day")
    ref_low_active: float = Field(..., gt=0, description="Reference low for the day")
    max_favorable_excursion_price: float = Field(..., gt=0, description="Best price reached during trade")
    min_adverse_excursion_price: float = Field(..., gt=0, description="Worst price reached during trade")
    max_profit_before_close: float = Field(default=0.0, ge=0, description="Maximum profit before closing")
    pnl: Optional[float] = Field(default=None, description="Profit and loss")
    exit_price: Optional[float] = Field(default=None, gt=0, description="Exit price")
    exit_time: Optional[datetime] = Field(default=None, description="Exit timestamp")
    
    @validator('day_date_str')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('day_date_str must be in YYYY-MM-DD format')
    
    @validator('exit_time')
    def validate_exit_after_entry(cls, v, values):
        if v is not None and 'entry_time' in values and values['entry_time'] is not None:
            if v < values['entry_time']:
                raise ValueError('exit_time must be after entry_time')
        return v

def create_trade(
    day_date_str: str,
    trade_num_day: int,
    trade_type: TradeType,
    entry_time: datetime,
    entry_price: float,
    stop_price: float,
    units: float,
    position_value_usd: float,
    ref_high_active: float,
    ref_low_active: float
) -> Trade:
    """Factory function to create a new Trade with validation."""
    return Trade(
        day_date_str=day_date_str,
        trade_num_day=trade_num_day,
        type=trade_type,
        entry_time=entry_time,
        entry_price=entry_price,
        stop_price=stop_price,
        units=units,
        position_value_usd=position_value_usd,
        status=TradeStatus.open,
        ref_high_active=ref_high_active,
        ref_low_active=ref_low_active,
        max_favorable_excursion_price=entry_price,
        min_adverse_excursion_price=entry_price,
        max_profit_before_close=0.0
    )
