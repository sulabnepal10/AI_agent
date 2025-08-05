from pydantic import BaseModel, Field

class StartupData(BaseModel):
    url: str = Field(..., description="The website URL of the startup.")
    name: str = Field(..., description="The name of the startup company.")
    founders: list[str] = Field(default=[], description="List of founder names.")
    emails: list[str] = Field(default=[], description="List of email addresses.")
    pages_crawled: int = Field(default=0, description="Number of pages successfully crawled.")
    confidence_score: float = Field(default=0.0, description="Confidence in extraction quality.")