"""
Memory module for retrieving client history and context.
Provides AI with relevant past interactions for personalization.
"""

from typing import Optional
from datetime import datetime

from database import get_db, Database
from models import ClientHistory, Client, Interaction


class MemoryManager:
    """
    Manages client context and conversation history.
    Retrieves relevant past interactions for AI context windows.
    """

    def __init__(self, db: Database = None):
        self.db = db or get_db()

    def get_client_history(self, client_id: int) -> ClientHistory:
        """
        Retrieve complete client history including all interactions.
        Returns structured data for AI context injection.
        """
        # Get client info
        client = self.db.get_client(client_id)
        if not client:
            raise ValueError(f"Client {client_id} not found")

        # Get all interactions
        interactions = self.db.get_client_interactions(client_id)

        # Calculate metadata
        total = len(interactions)
        last_contact = interactions[0].date if interactions else None

        return ClientHistory(
            client=client,
            interactions=interactions,
            total_interactions=total,
            last_contact=last_contact
        )

    def get_context_for_ai(self, client_id: int, max_interactions: int = 3) -> str:
        """
        Generate condensed context string for AI prompts.
        Limits to recent interactions to manage token usage.
        """
        try:
            history = self.get_client_history(client_id)

            # Limit interactions for context window management
            recent = history.interactions[:max_interactions]
            history.interactions = recent
            history.total_interactions = len(recent)

            return history.to_context_string()
        except ValueError:
            return "New client - no previous history."

    def get_client_timeline(self, client_id: int) -> list:
        """
        Get formatted timeline for UI display.
        Returns list of dicts for Streamlit timeline component.
        """
        history = self.get_client_history(client_id)
        timeline = []

        for inter in reversed(history.interactions):  # Oldest first
            timeline.append({
                'date': inter.date,
                'stage': inter.deal_stage,
                'summary': inter.summary,
                'interest': inter.interest_level,
                'next_action': inter.next_action
            })

        return timeline

    def get_similar_interactions(self, client_id: int, deal_stage: str) -> list:
        """
        Find past interactions in same stage for pattern matching.
        Useful for AI to learn from similar situations.
        """
        history = self.get_client_history(client_id)
        return [i for i in history.interactions if i.deal_stage == deal_stage]


def get_memory_manager() -> MemoryManager:
    """Factory function for memory manager."""
    return MemoryManager()