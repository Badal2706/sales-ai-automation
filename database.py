"""
Database layer with SQLite.
Handles connection pooling, schema creation, and CRUD operations.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager
import json
from difflib import SequenceMatcher

from config import DB_PATH, ensure_directories, DUPLICATE_SIMILARITY_THRESHOLD
from models import Client, ClientCreate, Interaction, InteractionCreate, FollowUp

class DatabaseError(Exception):
    """Custom database exception."""
    pass

class DuplicateClientError(Exception):
    """Raised when potential duplicate client detected."""
    pass

class Database:
    """
    SQLite database manager with connection pooling.
    Thread-safe for Streamlit usage.
    """

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        ensure_directories()
        self._init_database()
        self._run_migrations()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Database error: {str(e)}")
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database schema if not exists."""
        schema = """
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            deal_stage TEXT NOT NULL,
            objections TEXT,
            interest_level TEXT NOT NULL,
            next_action TEXT NOT NULL,
            followup_date TEXT,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS followups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            email_text TEXT NOT NULL,
            message_text TEXT NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_interactions_client ON interactions(client_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(date);
        CREATE INDEX IF NOT EXISTS idx_followups_interaction ON followups(interaction_id);
        CREATE INDEX IF NOT EXISTS idx_clients_name ON clients(name);
        CREATE INDEX IF NOT EXISTS idx_clients_email ON clients(email);
        """

        with self._get_connection() as conn:
            conn.executescript(schema)

    def _run_migrations(self) -> None:
        """Run database migrations for schema updates."""
        with self._get_connection() as conn:
            # Check if is_active column exists
            cursor = conn.execute("PRAGMA table_info(clients)")
            columns = [row['name'] for row in cursor.fetchall()]

            # Migration: Add is_active column if missing
            if 'is_active' not in columns:
                print("🔄 Running migration: Adding is_active column...")
                conn.execute("ALTER TABLE clients ADD COLUMN is_active BOOLEAN DEFAULT 1")
                print("✅ Migration complete")

            # Check if schema_version table exists (for future migrations)
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            if not cursor.fetchone():
                conn.execute("""
                    CREATE TABLE schema_version (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("INSERT INTO schema_version (version) VALUES ('1.0.1')")

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity percentage."""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() * 100

    def find_potential_duplicates(self, name: str, email: str = None,
                                  company: str = None) -> List[Dict[str, Any]]:
        """
        Find potential duplicate clients based on name/email similarity.
        Returns list of potential matches with similarity scores.
        """
        duplicates = []

        with self._get_connection() as conn:
            # Get all active clients
            rows = conn.execute(
                "SELECT * FROM clients WHERE is_active = 1 OR is_active IS NULL"
            ).fetchall()

            for row in rows:
                row_dict = dict(row)
                # Handle NULL is_active (treat as active)
                is_active = row_dict.get('is_active')
                if is_active is None:
                    is_active = 1

                if not is_active:
                    continue

                scores = {
                    'id': row_dict['id'],
                    'name': row_dict['name'],
                    'company': row_dict['company'],
                    'email': row_dict['email'],
                    'name_similarity': 0,
                    'email_match': False,
                    'company_similarity': 0,
                    'total_score': 0
                }

                # Check name similarity
                scores['name_similarity'] = self._calculate_similarity(name, row_dict['name'])

                # Check email exact match
                if email and row_dict['email'] and email.lower() == row_dict['email'].lower():
                    scores['email_match'] = True
                    scores['total_score'] = 100  # Exact email match = definite duplicate
                elif email and row_dict['email']:
                    scores['email_similarity'] = self._calculate_similarity(email, row_dict['email'])

                # Check company similarity
                if company and row_dict['company']:
                    scores['company_similarity'] = self._calculate_similarity(company, row_dict['company'])

                # Calculate total score (weighted)
                if not scores['email_match']:
                    scores['total_score'] = (
                        scores['name_similarity'] * 0.7 +
                        scores['company_similarity'] * 0.3
                    )

                # Check against threshold
                if scores['total_score'] >= DUPLICATE_SIMILARITY_THRESHOLD or scores['email_match']:
                    duplicates.append(scores)

        # Sort by total score descending
        return sorted(duplicates, key=lambda x: x['total_score'], reverse=True)

    # Client Operations

    def create_client(self, client: ClientCreate, force: bool = False) -> Client:
        """
        Create new client with duplicate detection.

        Args:
            client: Client data
            force: If True, skip duplicate check

        Raises:
            DuplicateClientError: If potential duplicate found and force=False
        """
        if not force:
            # Check for duplicates
            duplicates = self.find_potential_duplicates(
                client.name, client.email, client.company
            )

            if duplicates:
                raise DuplicateClientError(
                    f"Found {len(duplicates)} potential duplicate(s)",
                    duplicates
                )

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO clients (name, company, email, is_active) 
                   VALUES (?, ?, ?, 1)""",
                (client.name, client.company, client.email)
            )
            client_id = cursor.lastrowid

            # Fetch created client
            row = conn.execute(
                "SELECT * FROM clients WHERE id = ?", (client_id,)
            ).fetchone()

            return Client(**dict(row))

    def get_client(self, client_id: int) -> Optional[Client]:
        """Get client by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM clients 
                   WHERE id = ? 
                   AND (is_active = 1 OR is_active IS NULL)""",
                (client_id,)
            ).fetchone()

            return Client(**dict(row)) if row else None

    def get_all_clients(self, include_inactive: bool = False) -> List[Client]:
        """Get all clients ordered by creation date."""
        with self._get_connection() as conn:
            query = "SELECT * FROM clients"
            if not include_inactive:
                query += " WHERE is_active = 1 OR is_active IS NULL"
            query += " ORDER BY created_at DESC"

            rows = conn.execute(query).fetchall()

            return [Client(**dict(row)) for row in rows]

    def search_clients(self, query: str, include_inactive: bool = False) -> List[Client]:
        """Search clients by name or company."""
        with self._get_connection() as conn:
            pattern = f"%{query}%"
            sql = """SELECT * FROM clients 
                   WHERE (name LIKE ? OR company LIKE ?)"""
            if not include_inactive:
                sql += " AND (is_active = 1 OR is_active IS NULL)"
            sql += " ORDER BY name"

            rows = conn.execute(sql, (pattern, pattern)).fetchall()

            return [Client(**dict(row)) for row in rows]

    def update_client(self, client_id: int, **updates) -> Optional[Client]:
        """Update client fields."""
        allowed_fields = {'name', 'company', 'email'}
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if not update_fields:
            return None

        with self._get_connection() as conn:
            set_clause = ", ".join(f"{k} = ?" for k in update_fields)
            values = list(update_fields.values()) + [client_id]

            conn.execute(
                f"UPDATE clients SET {set_clause} WHERE id = ?",
                values
            )

            return self.get_client(client_id)

    def delete_client(self, client_id: int, soft_delete: bool = True) -> bool:
        """
        Delete client and all associated data.

        Args:
            client_id: Client to delete
            soft_delete: If True, mark as inactive. If False, permanent delete.
        """
        with self._get_connection() as conn:
            if soft_delete:
                # Soft delete - mark as inactive
                cursor = conn.execute(
                    "UPDATE clients SET is_active = 0 WHERE id = ?",
                    (client_id,)
                )
            else:
                # Hard delete - cascade will handle interactions/followups
                cursor = conn.execute(
                    "DELETE FROM clients WHERE id = ?",
                    (client_id,)
                )

            return cursor.rowcount > 0

    def restore_client(self, client_id: int) -> bool:
        """Restore a soft-deleted client."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE clients SET is_active = 1 WHERE id = ?",
                (client_id,)
            )
            return cursor.rowcount > 0

    def get_client_stats(self, client_id: int) -> Dict[str, Any]:
        """Get interaction stats for a client."""
        with self._get_connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_interactions,
                    MAX(date) as last_contact,
                    MIN(date) as first_contact,
                    GROUP_CONCAT(DISTINCT deal_stage) as stages_seen
                FROM interactions 
                WHERE client_id = ?
            """, (client_id,)).fetchone()

            return {
                'total_interactions': stats['total_interactions'],
                'last_contact': stats['last_contact'],
                'first_contact': stats['first_contact'],
                'stages_seen': stats['stages_seen'].split(',') if stats['stages_seen'] else []
            }

    # Interaction Operations

    def create_interaction(self, interaction: InteractionCreate) -> Interaction:
        """Create new interaction."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO interactions 
                   (client_id, raw_text, summary, deal_stage, objections, 
                    interest_level, next_action, followup_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (interaction.client_id, interaction.raw_text, interaction.summary,
                 interaction.deal_stage, interaction.objections, interaction.interest_level,
                 interaction.next_action, interaction.followup_date)
            )
            interaction_id = cursor.lastrowid

            row = conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()

            return Interaction(**dict(row))

    def get_interaction(self, interaction_id: int) -> Optional[Interaction]:
        """Get interaction by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()

            return Interaction(**dict(row)) if row else None

    def get_client_interactions(self, client_id: int) -> List[Interaction]:
        """Get all interactions for a client."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM interactions 
                   WHERE client_id = ? 
                   ORDER BY date DESC""",
                (client_id,)
            ).fetchall()

            return [Interaction(**dict(row)) for row in rows]

    def delete_interaction(self, interaction_id: int) -> bool:
        """Delete specific interaction and its followups."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM interactions WHERE id = ?", (interaction_id,)
            )
            return cursor.rowcount > 0

    def get_recent_interactions(self, limit: int = 10) -> List[Tuple[Interaction, Client]]:
        """Get recent interactions with client info."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT i.*, c.name as client_name, c.company 
                   FROM interactions i
                   JOIN clients c ON i.client_id = c.id
                   WHERE c.is_active = 1 OR c.is_active IS NULL
                   ORDER BY i.date DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()

            results = []
            for row in rows:
                row_dict = dict(row)
                client = Client(
                    id=row_dict['client_id'],
                    name=row_dict['client_name'],
                    company=row_dict['company'],
                    email=None,
                    created_at=datetime.now()
                )
                interaction = Interaction(
                    id=row_dict['id'],
                    client_id=row_dict['client_id'],
                    date=row_dict['date'],
                    raw_text=row_dict['raw_text'],
                    summary=row_dict['summary'],
                    deal_stage=row_dict['deal_stage'],
                    objections=row_dict['objections'],
                    interest_level=row_dict['interest_level'],
                    next_action=row_dict['next_action'],
                    followup_date=row_dict['followup_date']
                )
                results.append((interaction, client))

            return results

    # Follow-up Operations

    def create_followup(self, interaction_id: int, email: str, message: str) -> FollowUp:
        """Store generated follow-up content."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO followups (interaction_id, email_text, message_text)
                   VALUES (?, ?, ?)""",
                (interaction_id, email, message)
            )
            followup_id = cursor.lastrowid

            row = conn.execute(
                "SELECT * FROM followups WHERE id = ?", (followup_id,)
            ).fetchone()

            return FollowUp(**dict(row))

    def get_followup(self, interaction_id: int) -> Optional[FollowUp]:
        """Get follow-up by interaction ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM followups WHERE interaction_id = ?",
                (interaction_id,)
            ).fetchone()

            return FollowUp(**dict(row)) if row else None

    def get_all_followups(self, include_inactive: bool = False) -> List[Tuple[FollowUp, Interaction, Client]]:
        """Get all follow-ups with context."""
        with self._get_connection() as conn:
            sql = """SELECT f.*, i.summary, i.deal_stage, i.interest_level, 
                          i.next_action, i.date, c.name, c.company, c.id as client_id
                   FROM followups f
                   JOIN interactions i ON f.interaction_id = i.id
                   JOIN clients c ON i.client_id = c.id"""

            if not include_inactive:
                sql += " WHERE (c.is_active = 1 OR c.is_active IS NULL)"

            sql += " ORDER BY i.date DESC"

            rows = conn.execute(sql).fetchall()

            results = []
            for row in rows:
                row_dict = dict(row)
                followup = FollowUp(
                    id=row_dict['id'],
                    interaction_id=row_dict['interaction_id'],
                    email_text=row_dict['email_text'],
                    message_text=row_dict['message_text']
                )
                interaction = Interaction(
                    id=row_dict['interaction_id'],
                    client_id=row_dict['client_id'],
                    date=row_dict['date'],
                    raw_text="",
                    summary=row_dict['summary'],
                    deal_stage=row_dict['deal_stage'],
                    objections=None,
                    interest_level=row_dict['interest_level'],
                    next_action=row_dict['next_action'],
                    followup_date=None
                )
                client = Client(
                    id=row_dict['client_id'],
                    name=row_dict['name'],
                    company=row_dict['company'],
                    email=None,
                    created_at=datetime.now()
                )
                results.append((followup, interaction, client))

            return results

    # Analytics

    def get_pipeline_stats(self, include_inactive: bool = False) -> Dict[str, int]:
        """Get deal stage counts."""
        with self._get_connection() as conn:
            sql = """SELECT deal_stage, COUNT(*) as count 
                   FROM interactions i
                   JOIN clients c ON i.client_id = c.id"""

            if not include_inactive:
                sql += " WHERE (c.is_active = 1 OR c.is_active IS NULL)"

            sql += " GROUP BY deal_stage"

            rows = conn.execute(sql).fetchall()

            return {row['deal_stage']: row['count'] for row in rows}

    def get_interactions_needing_followup(self, days: int = 1) -> List[Tuple[Interaction, Client]]:
        """Get interactions where followup_date is due."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT i.*, c.name, c.company, c.email
                   FROM interactions i
                   JOIN clients c ON i.client_id = c.id
                   WHERE (c.is_active = 1 OR c.is_active IS NULL)
                   AND i.followup_date <= date('now', '+{} days')
                   AND i.followup_date >= date('now')
                   AND NOT EXISTS (
                       SELECT 1 FROM followups f WHERE f.interaction_id = i.id
                   )
                   ORDER BY i.followup_date""".format(days)
            ).fetchall()

            results = []
            for row in rows:
                row_dict = dict(row)
                client = Client(
                    id=row_dict['client_id'],
                    name=row_dict['name'],
                    company=row_dict['company'],
                    email=row_dict['email'],
                    created_at=datetime.now()
                )
                interaction = Interaction(
                    id=row_dict['id'],
                    client_id=row_dict['client_id'],
                    date=row_dict['date'],
                    raw_text=row_dict['raw_text'],
                    summary=row_dict['summary'],
                    deal_stage=row_dict['deal_stage'],
                    objections=row_dict['objections'],
                    interest_level=row_dict['interest_level'],
                    next_action=row_dict['next_action'],
                    followup_date=row_dict['followup_date']
                )
                results.append((interaction, client))

            return results

# Singleton instance for app usage
_db_instance = None

def get_db() -> Database:
    """Get or create database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance