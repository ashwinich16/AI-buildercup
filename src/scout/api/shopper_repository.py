"""
What Scout keeps in the database, behind two small interfaces:

  ShopperRepository   what Scout remembers about a shopper between conversations: the name they asked to
                      be called, their preferences and taste, their cart, their wishlist and their wardrobe
  ConversationStore   the current conversation, as one compact summary row (Scout's session state: what is
                      being looked for, the options shown, the outfit plan, the last few exchanges), never
                      the full message history. It lets a conversation carry on after a server restart or
                      on another Cloud Run instance. Rows untouched for a week are deleted.

Both are kept here so they share one database connection. Two interchangeable implementations:
  CloudSqlShopperRepository  - Cloud SQL (PostgreSQL); used when SCOUT_DB_INSTANCE is set
  InMemoryShopperRepository  - in memory, for running without a database (lost on restart)

The tables are created automatically if they don't exist, so a fresh database needs no setup.
Preferences are stored as rows (attribute / value), never as one column per kind of
preference, so the tables don't depend on what the catalog sells.
"""

import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

from pydantic import BaseModel, Field

from scout.config import Settings
from scout.domain import Preference, PreferenceProfile, TasteProfile, WardrobeItem
from scout.domain.taste import TasteMark
from scout.tools import Cart, CartItem, Wishlist, WishlistItem

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10      # opening a database connection; normally well under a second in the same region


class ShopperMemory(BaseModel):
    """Everything Scout remembers about one shopper."""
    known: bool = False                    # True if they have shopped with Scout before
    preferred_name: str = ""               # what they asked Scout to call them
    preferences: PreferenceProfile = Field(default_factory=PreferenceProfile)
    cart: Cart = Field(default_factory=Cart)
    wishlist: Wishlist = Field(default_factory=Wishlist)   # products saved for later
    taste: TasteProfile = Field(default_factory=TasteProfile)   # learned from what they own and shop
    wardrobe: list[WardrobeItem] = Field(default_factory=list)   # clothes they already own


class ShopperRepository(ABC):
    @abstractmethod
    def load(self, user_id: str) -> ShopperMemory:
        """What is remembered about this shopper (empty for a new shopper)."""

    @abstractmethod
    def save(self, user_id: str, memory: ShopperMemory) -> None:
        """Remember this as the shopper's current state."""


class ConversationStore(ABC):
    @abstractmethod
    def save_conversation(self, user_id: str, session_id: str, summary: dict) -> None:
        """Keep the conversation's summary (replacing the previous one)."""

    @abstractmethod
    def load_conversation(self, user_id: str, session_id: str) -> Optional[dict]:
        """This shopper's conversation summary, or None if there is none."""

    @abstractmethod
    def delete_conversation(self, user_id: str, session_id: str) -> None:
        """Forget the conversation (the shopper cleared it)."""

    @abstractmethod
    def latest_conversation(self, user_id: str) -> Optional[str]:
        """The id of this shopper's most recent conversation, to carry on where they left off."""


class InMemoryShopperRepository(ShopperRepository, ConversationStore):
    def __init__(self):
        self._shoppers: dict[str, ShopperMemory] = {}
        self._conversations: dict[tuple[str, str], str] = {}   # in the order they were last saved

    def load(self, user_id: str) -> ShopperMemory:
        memory = self._shoppers.get(user_id)
        return memory.model_copy(deep=True) if memory else ShopperMemory()

    def save(self, user_id: str, memory: ShopperMemory) -> None:
        self._shoppers[user_id] = memory.model_copy(deep=True, update={"known": True})

    def save_conversation(self, user_id: str, session_id: str, summary: dict) -> None:
        self._conversations.pop((user_id, session_id), None)          # so the newest is always last
        self._conversations[(user_id, session_id)] = json.dumps(summary, default=str)

    def load_conversation(self, user_id: str, session_id: str) -> Optional[dict]:
        saved = self._conversations.get((user_id, session_id))
        return json.loads(saved) if saved else None

    def delete_conversation(self, user_id: str, session_id: str) -> None:
        self._conversations.pop((user_id, session_id), None)

    def latest_conversation(self, user_id: str) -> Optional[str]:
        return next((sid for uid, sid in reversed(self._conversations) if uid == user_id), None)


class CloudSqlShopperRepository(ShopperRepository, ConversationStore):
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
        "ALTER TABLE cart_item ADD COLUMN IF NOT EXISTS kind TEXT",   # added later; older tables lack it
        "ALTER TABLE cart_item ADD COLUMN IF NOT EXISTS image_url TEXT",
        "ALTER TABLE cart_item ADD COLUMN IF NOT EXISTS size TEXT",
        """CREATE TABLE IF NOT EXISTS wishlist_item (
            user_id     TEXT NOT NULL,
            product_id  TEXT NOT NULL,
            name        TEXT NOT NULL,
            price       DOUBLE PRECISION,
            kind        TEXT,
            image_url   TEXT,
            position    INTEGER NOT NULL,
            added_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, product_id)
        )""",
        """CREATE TABLE IF NOT EXISTS taste_mark (
            user_id     TEXT NOT NULL,
            trait       TEXT NOT NULL,
            value       TEXT NOT NULL,
            style       TEXT NOT NULL,
            weight      DOUBLE PRECISION NOT NULL,
            items       INTEGER NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, trait, value, style)
        )""",
        """CREATE TABLE IF NOT EXISTS wardrobe_item (
            user_id     TEXT NOT NULL,
            item_id     TEXT NOT NULL,
            category    TEXT NOT NULL,
            description TEXT NOT NULL,
            thumbnail   TEXT,
            added_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, item_id)
        )""",
        """CREATE TABLE IF NOT EXISTS conversation (
            user_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            summary     TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, session_id)
        )""",
        # Conversations nobody came back to within a week are not worth keeping.
        "DELETE FROM conversation WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '7 days'",
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
            "SELECT product_id, name, price, kind, image_url, size FROM cart_item WHERE user_id = %s "
            "ORDER BY position", (user_id,))
        items = [CartItem(product_id=pid, name=name, price=None if price is None else float(price), kind=kind,
                          image_url=image, size=size)
                 for pid, name, price, kind, image, size in cursor.fetchall()]

        cursor.execute("SELECT product_id, name, price, kind, image_url FROM wishlist_item WHERE user_id = %s "
                       "ORDER BY position", (user_id,))
        saved = [WishlistItem(product_id=pid, name=name, price=None if price is None else float(price), kind=kind,
                              image_url=image)
                 for pid, name, price, kind, image in cursor.fetchall()]

        cursor.execute("SELECT trait, value, style, weight, items FROM taste_mark WHERE user_id = %s "
                       "ORDER BY weight DESC", (user_id,))
        marks = [TasteMark(trait=t, value=v, style=s, weight=float(w), items=n) for t, v, s, w, n in cursor.fetchall()]

        cursor.execute("SELECT item_id, category, description, thumbnail FROM wardrobe_item "
                       "WHERE user_id = %s ORDER BY added_at", (user_id,))
        wardrobe = [WardrobeItem(id=i, category=c, description=d, thumbnail=t) for i, c, d, t in cursor.fetchall()]

        return ShopperMemory(
            known=True,
            preferred_name=shopper[0],
            preferences=PreferenceProfile(preferences=preferences),
            cart=Cart(items=items),
            wishlist=Wishlist(items=saved),
            taste=TasteProfile(marks=marks),
            wardrobe=wardrobe,
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
        self._save_wishlist(cursor, user_id, memory.wishlist.items)
        self._save_taste(cursor, user_id, memory.taste.marks)
        self._save_wardrobe(cursor, user_id, memory.wardrobe)

    @staticmethod
    def _save_wardrobe(cursor, user_id: str, items: list[WardrobeItem]) -> None:
        wanted = {item.id for item in items}
        cursor.execute("SELECT item_id FROM wardrobe_item WHERE user_id = %s", (user_id,))
        for (item_id,) in cursor.fetchall():
            if item_id not in wanted:                       # taken out of the wardrobe
                cursor.execute("DELETE FROM wardrobe_item WHERE user_id = %s AND item_id = %s", (user_id, item_id))
        for item in items:
            cursor.execute(
                "INSERT INTO wardrobe_item (user_id, item_id, category, description, thumbnail) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (user_id, item_id) DO UPDATE SET "
                "category = EXCLUDED.category, description = EXCLUDED.description",
                (user_id, item.id, item.category, item.description, item.thumbnail))

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
                "INSERT INTO cart_item (user_id, product_id, name, price, kind, image_url, size, position) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, product_id) DO UPDATE SET "
                "name = EXCLUDED.name, price = EXCLUDED.price, kind = EXCLUDED.kind, "
                "image_url = COALESCE(EXCLUDED.image_url, cart_item.image_url), size = EXCLUDED.size, "
                "position = EXCLUDED.position",
                (user_id, item.product_id, item.name, item.price, item.kind, item.image_url, item.size, position))

    @staticmethod
    def _save_wishlist(cursor, user_id: str, items: list[WishlistItem]) -> None:
        wanted = {item.product_id for item in items}
        cursor.execute("SELECT product_id FROM wishlist_item WHERE user_id = %s", (user_id,))
        for (product_id,) in cursor.fetchall():
            if product_id not in wanted:                    # taken off the wishlist
                cursor.execute("DELETE FROM wishlist_item WHERE user_id = %s AND product_id = %s",
                               (user_id, product_id))
        for position, item in enumerate(items, start=1):
            cursor.execute(
                "INSERT INTO wishlist_item (user_id, product_id, name, price, kind, image_url, position) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, product_id) DO UPDATE SET "
                "name = EXCLUDED.name, price = EXCLUDED.price, kind = EXCLUDED.kind, "
                "image_url = COALESCE(EXCLUDED.image_url, wishlist_item.image_url), position = EXCLUDED.position",
                (user_id, item.product_id, item.name, item.price, item.kind, item.image_url, position))

    @staticmethod
    def _save_taste(cursor, user_id: str, marks: list[TasteMark]) -> None:
        wanted = {mark.key(): mark for mark in marks}
        cursor.execute("SELECT trait, value, style FROM taste_mark WHERE user_id = %s", (user_id,))
        for key in {tuple(row) for row in cursor.fetchall()} - set(wanted):     # forgotten, or dropped as lightest
            cursor.execute("DELETE FROM taste_mark WHERE user_id = %s AND trait = %s AND value = %s AND style = %s",
                           (user_id, *key))
        for mark in marks:
            cursor.execute(
                "INSERT INTO taste_mark (user_id, trait, value, style, weight, items) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, trait, value, style) DO UPDATE SET weight = EXCLUDED.weight, "
                "items = EXCLUDED.items, updated_at = CURRENT_TIMESTAMP",
                (user_id, mark.trait, mark.value, mark.style, mark.weight, mark.items))

    # ----- the current conversation -----

    def save_conversation(self, user_id: str, session_id: str, summary: dict) -> None:
        self._run(lambda cursor: cursor.execute(
            "INSERT INTO conversation (user_id, session_id, summary) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, session_id) DO UPDATE SET summary = EXCLUDED.summary, "
            "updated_at = CURRENT_TIMESTAMP",
            (user_id, session_id, json.dumps(summary, default=str))))

    def load_conversation(self, user_id: str, session_id: str) -> Optional[dict]:
        def load(cursor):
            cursor.execute("SELECT summary FROM conversation WHERE user_id = %s AND session_id = %s",
                           (user_id, session_id))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
        return self._run(load)

    def delete_conversation(self, user_id: str, session_id: str) -> None:
        self._run(lambda cursor: cursor.execute(
            "DELETE FROM conversation WHERE user_id = %s AND session_id = %s", (user_id, session_id)))

    def latest_conversation(self, user_id: str) -> Optional[str]:
        def latest(cursor):
            cursor.execute("SELECT session_id FROM conversation WHERE user_id = %s "
                           "ORDER BY updated_at DESC LIMIT 1", (user_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        return self._run(latest)

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


def build_shopper_repository(settings: Settings) -> CloudSqlShopperRepository | InMemoryShopperRepository:
    """One object for both interfaces (ShopperRepository and ConversationStore), sharing the connection."""
    if not settings.db_instance:
        logger.warning("SCOUT_DB_INSTANCE not set: names, preferences and carts live in memory only.")
        return InMemoryShopperRepository()

    from google.cloud import secretmanager                 # only needed with Cloud SQL
    from google.cloud.sql.connector import Connector

    secret = (f"projects/{settings.project_id}/secrets/{settings.db_password_secret}/versions/latest")
    connector = None
    password = None

    def connect():
        """Called on first use and on reconnects only, never at server start.

        "lazy": certificates are fetched when a connection is opened, not refreshed in the background (Cloud Run
        may pause a server's CPU between requests, which stalls background refreshes). The timeout means a
        database that cannot be reached fails in seconds instead of holding every request up."""
        nonlocal connector, password
        connector = connector or Connector(refresh_strategy="lazy")
        password = password or secretmanager.SecretManagerServiceClient().access_secret_version(
            name=secret).payload.data.decode("utf-8").strip()
        return connector.connect(settings.db_instance, "pg8000", user=settings.db_user,
                                 password=password, db=settings.db_name, timeout=CONNECT_TIMEOUT_S)

    return CloudSqlShopperRepository(connect)
