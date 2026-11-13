"""Request schemas for manually submitted section summaries."""

from pydantic import BaseModel, Field


class SectionSummaryRequest(BaseModel):
    summary: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Manually generated summary of the research-paper section",
    )