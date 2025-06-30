from typing import List, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from models.trade import Trade

class DailyResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    Day: str = Field(..., description="Date string in YYYY-MM-DD format")
    Ref_High: float = Field(..., gt=0, description="Reference high for the day")
    Ref_Low: float = Field(..., gt=0, description="Reference low for the day")
    Trades_Info: List["Trade"] = Field(default_factory=list, description="List of trades for the day")

# Update forward references after all models are defined
def update_forward_refs():
    from models.trade import Trade
    DailyResult.model_rebuild()