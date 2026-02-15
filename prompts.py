"""
All AI prompts centralized for easy tuning and versioning.
No hardcoded prompts in business logic.
"""

from typing import Dict

class Prompts:
    """
    Centralized prompt templates.
    Use .format() or f-strings for variable injection.
    """

    CRM_EXTRACTION = """You are an expert sales CRM assistant. Analyze the following sales conversation and extract structured data.

CONVERSATION:
{conversation}

CLIENT CONTEXT:
{context}

Extract the following fields and return ONLY a valid JSON object (no markdown, no explanation):

{{
    "summary": "Brief 2-3 sentence summary of what was discussed and key outcomes",
    "deal_stage": "One of: prospecting, qualification, proposal, negotiation, closed_won, closed_lost, nurture",
    "objections": "Any concerns or objections raised by client, or null if none",
    "interest_level": "One of: hot, warm, cold, neutral",
    "next_action": "Specific next step required (e.g., 'Send proposal by Friday', 'Schedule demo next Tuesday')",
    "followup_date": "Suggested follow-up date in YYYY-MM-DD format, or null if not applicable"
}}

Rules:
- Be concise but specific in summaries
- Deal stage must be exact match from list
- Interest level: hot=ready to buy, warm=interested, cold=not interested, neutral=unclear
- Next action must be actionable and specific
- If client mentioned specific dates/times, use those for followup_date"""

    EMAIL_FOLLOWUP = """You are a professional sales copywriter. Write a follow-up email based on the interaction details.

CLIENT: {client_name}
COMPANY: {company}
HISTORY:
{history}

CURRENT INTERACTION:
{summary}
Deal Stage: {deal_stage}
Interest Level: {interest_level}
Next Action: {next_action}
Objections: {objections}

Write a professional, personalized follow-up email that:
1. References specific points from the conversation
2. Addresses any objections if present
3. Confirms the next action
4. Maintains appropriate tone for the interest level (hot=urgent, warm=friendly, cold=gentle, neutral=professional)
5. Is 3-5 paragraphs max
6. Includes professional signature

Return ONLY the email body text, no subject line, no markdown formatting."""

    MESSAGE_FOLLOWUP = """You are a sales assistant writing a WhatsApp/SMS follow-up. Create a short, casual message.

CLIENT: {client_name}
CONTEXT: {summary}
NEXT ACTION: {next_action}
INTEREST LEVEL: {interest_level}

Write a brief WhatsApp-style message (2-4 sentences) that:
- Is conversational and friendly
- References the discussion
- Confirms next steps
- Uses appropriate urgency based on interest level
- No formal salutation or signature needed
- Under 300 characters if possible, max 500

Return ONLY the message text."""

    SYSTEM_PROMPT = """You are a professional Sales AI Assistant. Your tasks:
1. Extract structured CRM data from conversations
2. Generate contextual follow-up communications
3. Maintain professional, helpful tone
4. Always return valid, parseable output
5. Be concise but thorough"""

    @classmethod
    def get_crm_prompt(cls, conversation: str, context: str = "New client") -> str:
        """Generate CRM extraction prompt."""
        return cls.CRM_EXTRACTION.format(
            conversation=conversation,
            context=context
        )

    @classmethod
    def get_email_prompt(cls, client_name: str, company: str, history: str,
                        summary: str, deal_stage: str, interest_level: str,
                        next_action: str, objections: str = None) -> str:
        """Generate email follow-up prompt."""
        # Handle None objections
        objections_str = objections if objections and objections.strip() else "None"

        return cls.EMAIL_FOLLOWUP.format(
            client_name=client_name,
            company=company or "Unknown",
            history=history,
            summary=summary,
            deal_stage=deal_stage,
            interest_level=interest_level,
            next_action=next_action,
            objections=objections_str
        )

    @classmethod
    def get_message_prompt(cls, client_name: str, summary: str,
                          next_action: str, interest_level: str) -> str:
        """Generate WhatsApp message prompt."""
        return cls.MESSAGE_FOLLOWUP.format(
            client_name=client_name,
            summary=summary,
            next_action=next_action,
            interest_level=interest_level
        )