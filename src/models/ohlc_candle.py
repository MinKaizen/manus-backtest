from pydantic import BaseModel, Field, ConfigDict, validator

class OHLCCandle(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    
    @validator('high')
    def validate_high_is_highest(cls, v, values):
        if 'open' in values and v < values['open']:
            raise ValueError('high must be >= open')
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        if 'close' in values and v < values['close']:
            raise ValueError('high must be >= close')
        return v
    
    @validator('low')
    def validate_low_is_lowest(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('low must be <= close')
        return v