"""
Pydantic models for data validation and serialization.
Ensures type safety and JSON schema compliance.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator
import json


class DealStage(str, Enum):
    """Valid deal stages in sales pipeline."""
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    NURTURE = "nurture"


class InterestLevel(str, Enum):
    """Client interest temperature."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    NEUTRAL = "neutral"


class ClientBase(BaseModel):
    """Base client schema."""
    name: str = Field(..., min_length=1, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    email: Optional[str] = Field(None, max_length=100)


class ClientCreate(ClientBase):
    """Schema for creating new client."""
    pass


class Client(ClientBase):
    """Full client schema with DB fields."""
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class CRMData(BaseModel):
    """
    Structured output from CRM AI.
    Strict validation ensures downstream reliability.
    """
    summary: str = Field(..., min_length=10, description="Concise interaction summary")
    deal_stage: DealStage = Field(..., description="Current pipeline stage")
    objections: Optional[str] = Field(None, description="Client objections/concerns")
    interest_level: InterestLevel = Field(..., description="Temperature of lead")
    next_action: str = Field(..., min_length=5, description="Specific next step")
    followup_date: Optional[str] = Field(None, description="Suggested follow-up date (YYYY-MM-DD)")

    @validator('followup_date')
    def validate_date_format(cls, v):
        """Ensure date is valid YYYY-MM-DD format."""
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @validator('objections')
    def clean_objections(cls, v):
        """Clean empty objections to None."""
        if v and v.strip().lower() in ['none', 'n/a', 'no objections', '']:
            return None
        return v


class InteractionBase(BaseModel):
    """Base interaction schema."""
    client_id: int
    raw_text: str = Field(..., min_length=10, description="Original conversation text")


class InteractionCreate(InteractionBase):
    """Schema for creating interaction with AI data."""
    summary: str
    deal_stage: str
    objections: Optional[str]
    interest_level: str
    next_action: str
    followup_date: Optional[str]


class Interaction(InteractionCreate):
    """Full interaction schema."""
    id: int
    date: datetime

    class Config:
        orm_mode = True


class FollowUpContent(BaseModel):
    """Generated follow-up content."""
    email_text: str = Field(..., min_length=20, description="Professional email follow-up")
    message_text: str = Field(..., min_length=10, description="Short WhatsApp-style message")


class FollowUp(BaseModel):
    """Stored follow-up in database."""
    id: int
    interaction_id: int
    email_text: str
    message_text: str

    class Config:
        orm_mode = True


class ClientHistory(BaseModel):
    """Complete client context for AI."""
    client: Client
    interactions: List[Interaction]
    total_interactions: int
    last_contact: Optional[datetime]

    def to_context_string(self) -> str:
        """Convert history to string for AI context."""
        lines = [
            f"Client: {self.client.name} ({self.client.company or 'No company'})",
            f"Total interactions: {self.total_interactions}",
            f"Last contact: {self.last_contact.strftime('%Y-%m-%d') if self.last_contact else 'Never'}",
            "\nRecent History:"
        ]

        for idx, inter in enumerate(self.interactions[-3:], 1):
            lines.append(f"\n{idx}. {inter.date.strftime('%Y-%m-%d')} - {inter.deal_stage}")
            lines.append(f"   Summary: {inter.summary}")
            if inter.objections:
                lines.append(f"   Objections: {inter.objections}")

        return "\n".join(lines)


def validate_json_output(json_str: str) -> CRMData:
    """
    Validate and parse AI JSON output.
    Raises ValueError if invalid.
    """
    try:
        data = json.loads(json_str)
        return CRMData(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from AI: {str(e)}")
    except Exception as e:
        raise ValueError(f"Schema validation failed: {str(e)}")