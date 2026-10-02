"""
What Scout remembers about a shopper between conversations: the name they
asked to be called, their preferences and their cart.

The ADK session holds only the current conversation (what is being searched,
which options were just shown). Everything worth remembering lives here, so it
survives server restarts and is shared by every Cloud Run instance.

Two interchangeable implementations:
  CloudSqlShopperRepository  - Cloud SQL (PostgreSQL); used when SCOUT_DB_INSTANCE is set
  InMemoryShopperRepository  - in memory, for running without a database (lost on restart)

The tables are created automatically if they don't exist, so a fresh database needs no setup.
Preferences are stored as rows (attribute / value), never as one column per kind of
preference, so the tables don't depend on what the catalog sells.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, Field

from scout.config import Settings
from scout.domain import Preference, PreferenceProfile
from scout.tools import Cart, CartItem

logger = logging.getLogger(__name__)


class ShopperMemory(BaseModel):
    """Everything Scout remembers about one shopper."""
    known: bool = False                    # True if they have shopped with Scout before
    preferred_name: str = ""               # what they asked Scout to call them
    preferences: PreferenceProfile = Field(default_factory=PreferenceProfile)
    cart: Cart = Field(default_factory=Cart)


class ShopperRepository(ABC):
    @abstractmethod
    def load(self, user_id: str) -> ShopperMemory:
        """What is remembered about this shopper (empty for a new shopper)."""

    @abstractmethod
    def save(self, user_id: str, memory: ShopperMemory) -> None:
        """Remember this as the shopper's current state."""


class InMemoryShopperRepository(ShopperRepository):
    def __init__(self):
        self._shoppers: dict[str, ShopperMemory] = {}

    def load(self, user_id: str) -> ShopperMemory:
        memory = self._shoppers.get(user_id)
        return memory.model_copy(deep=True) if memory else ShopperMemory()

    def save(self, user_id: str, memory: ShopperMemory) -> None:
        self._shoppers[user_id] = memory.model_copy(deep=True, update={"known": True})


class CloudSqlShopperRepository(ShopperRepository):
    SCHEMA = [
        """CREATE TABLE IF NOT EXISTS shopper (
            user_id         TEXT PRIMARY KEY,
            preferred_name  TEXT NOT NULL DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS user_preference (
            user_id     TEXT NOT NULL,
            attribute   TEXT NOT NULL,
            value       TEXT NOT NULL,
            scope       TEXT NOT NULL DEFAULT '',
            polarity    TEXT NOT NULL CHECK (polarity IN ('like', 'avoid')),
            source      TEXT NOT NULL CHECK (source IN ('stated', 'learned')),
            confidence  DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            evidence    INTEGER NOT NULL DEFAULT 1,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, attribute, value, scope)
        )""",
        """CREATE TABLE IF NOT EXISTS cart_item (
            user_id     TEXT NOT NULL,
            product_id  TEXT NOT NULL,
            name        TEXT NOT NULL,
            price       DOUBLE PRECISION,
            position    INTEGER NOT NULL,
            added_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, product_id)
        )""",
    ]

    def __init__(self, connect: Callable[[], object]):
        """`connect` opens a DB-API connection (format paramstyle, e.g. pg8000)."""
        self._connect = connect
        self._connection = None
        self._tables_ready = False             # created on first use, so the server starts even if the DB is down
        self._lock = threading.Lock()          # one shared connection, used by one request at a time

    def _create_tables(self, cursor) -> None:
        for sql in self.SCHEMA:
            cursor.execute(sql)

    # ----- reading -----

    def load(self, user_id: str) -> ShopperMemory:
        return self._run(lambda cursor: self._load(cursor, user_id))

    @staticmethod
    def _load(cursor, user_id: str) -> ShopperMemory:
        cursor.execute("SELECT preferred_name FROM shopper WHERE user_id = %s", (user_id,))
        shopper = cursor.fetchone()
        if shopper is None:
            return ShopperMemory()

        cursor.execute(
            "SELECT attribute, value, scope, polarity, source, confidence, evidence "
            "FROM user_preference WHERE user_id = %s "
            "ORDER BY source = 'stated' DESC, confidence DESC", (user_id,))
        preferences = [
            Preference(attribute=a, value=v, scope=s or None, polarity=p, source=src,
                       confidence=float(c), evidence=e)
            for a, v, s, p, src, c, e in cursor.fetchall()
        ]
        cursor.execute(
            "SELECT product_id, name, price FROM cart_item WHERE user_id = %s ORDER BY position", (user_id,))
        items = [CartItem(product_id=pid, name=name, price=None if price is None else float(price))
                 for pid, name, price in cursor.fetchall()]

        return ShopperMemory(
            known=True,
            preferred_name=shopper[0],
            preferences=PreferenceProfile(preferences=preferences),
            cart=Cart(items=items),
        )

    # ----- writing -----

    def save(self, user_id: str, memory: ShopperMemory) -> None:
        self._run(lambda cursor: self._save(cursor, user_id, memory))

    def _save(self, cursor, user_id: str, memory: ShopperMemory) -> None:
        cursor.execute(
            "INSERT INTO shopper (user_id, preferred_name) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET preferred_name = EXCLUDED.preferred_name, "
            "last_seen_at = CURRENT_TIMESTAMP",
            (user_id, memory.preferred_name))
        self._save_preferences(cursor, user_id, memory.preferences.preferences)
        self._save_cart(cursor, user_id, memory.cart.items)

    @staticmethod
    def _save_preferences(cursor, user_id: str, preferences: list[Preference]) -> None:
        wanted = {p.key(): p for p in preferences}           # (attribute, value, scope), lowercased
        cursor.execute("SELECT attribute, value, scope FROM user_preference WHERE user_id = %s", (user_id,))
        for key in {tuple(row) for row in cursor.fetchall()} - set(wanted):   # forgotten ones
            cursor.execute(
                "DELETE FROM user_preference WHERE user_id = %s AND attribute = %s AND value = %s AND scope = %s",
                (user_id, *key))
        for (attribute, value, scope), p in wanted.items():
            cursor.execute(
                "INSERT INTO user_preference "
                "(user_id, attribute, value, scope, polarity, source, confidence, evidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, attribute, value, scope) DO UPDATE SET "
                "polarity = EXCLUDED.polarity, source = EXCLUDED.source, confidence = EXCLUDED.confidence, "
                "evidence = EXCLUDED.evidence, updated_at = CURRENT_TIMESTAMP",
                (user_id, attribute, value, scope, p.polarity, p.source, p.confidence, p.evidence))

    @staticmethod
    def _save_cart(cursor, user_id: str, items: list[CartItem]) -> None:
        wanted = {item.product_id for item in items}
        cursor.execute("SELECT product_id FROM cart_item WHERE user_id = %s", (user_id,))
        for (product_id,) in cursor.fetchall():
            if product_id not in wanted:                    # removed from the cart
                cursor.execute("DELETE FROM cart_item WHERE user_id = %s AND product_id = %s",
                               (user_id, product_id))
        for position, item in enumerate(items, start=1):
            cursor.execute(
                "INSERT INTO cart_item (user_id, product_id, name, price, position) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, product_id) DO UPDATE SET "
                "name = EXCLUDED.name, price = EXCLUDED.price, position = EXCLUDED.position",
                (user_id, item.product_id, item.name, item.price, position))

    # ----- connection handling -----

    def _run(self, work):
        """Run `work(cursor)` in one transaction. Reconnects once if the connection went stale."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._connection is None:
                        self._connection = self._connect()
                    cursor = self._connection.cursor()
                    if not self._tables_ready:
                        self._create_tables(cursor)
                        self._tables_ready = True
                    result = work(cursor)
                    self._connection.commit()
                    return result
                except Exception:
                    self._discard_connection()
                    if attempt == 2:
                        raise
                    logger.warning("Shopper database call failed; reconnecting and retrying once.")

    def _discard_connection(self) -> None:
        try:
            if self._connection is not None:
                self._connection.rollback()
                self._connection.close()
        except Exception:
            pass
        self._connection = None


def build_shopper_repository(settings: Settings) -> ShopperRepository:
    if not settings.db_instance:
        logger.warning("SCOUT_DB_INSTANCE not set: names, preferences and carts live in memory only.")
        return InMemoryShopperRepository()

    from google.cloud import secretmanager                 # only needed with Cloud SQL
    from google.cloud.sql.connector import Connector

    secret = (f"projects/{settings.project_id}/secrets/{settings.db_password_secret}/versions/latest")
    connector = None

    def connect():
        """Called on first use and on reconnects only, never at server start."""
        nonlocal connector
        connector = connector or Connector()
        password = secretmanager.SecretManagerServiceClient().access_secret_version(
            name=secret).payload.data.decode("utf-8").strip()
        return connector.connect(settings.db_instance, "pg8000", user=settings.db_user,
                                 password=password, db=settings.db_name)

    return CloudSqlShopperRepository(connect)
