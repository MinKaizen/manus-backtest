from pydantic import BaseModel, Field, ConfigDict

class BacktestConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    risk_amount_dollars: float = Field(..., gt=0, description="Risk amount per trade in dollars")
    output_csv_path: str = Field(..., description="Path for CSV output file")
    analysis_output_path: str = Field(..., description="Path for analysis output file")
    data_filepath: str = Field(..., description="Path to input data file")