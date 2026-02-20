"""
SQLite storage implementation for local development and testing.

Provides relational data storage using SQLite with zero configuration.
"""
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .schema import get_create_table_sql, TABLES
from .storage_manager import StorageConfig

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """SQLite storage backend

    Zero-configuration storage for local development and unit testing.
    Supports all 21 tables defined in the schema.

    Example:
        config = StorageConfig(sqlite_path="./data/experiments.db")
        storage = SQLiteStorage(config)
        storage.connect()
        storage.create_tables()

        # Insert data
        storage.insert('satellites', {'id': 'OPT-01', 'name': 'Satellite 1'})

        # Query data
        result = storage.fetch_one("SELECT * FROM satellites WHERE id = ?", ('OPT-01',))
    """

    def __init__(self, config: StorageConfig):
        """Initialize SQLite storage

        Args:
            config: Storage configuration
        """
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def connect(self) -> None:
        """Establish database connection"""
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Enable foreign keys
        self.cursor.execute("PRAGMA foreign_keys = ON")

        # Optimize performance
        self.cursor.execute("PRAGMA journal_mode = WAL")
        self.cursor.execute("PRAGMA synchronous = NORMAL")
        self.cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache

        logger.info(f"Connected to SQLite database: {self.config.sqlite_path}")

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("SQLite connection closed")

    def is_connected(self) -> bool:
        """Check if database connection is active"""
        return self.conn is not None

    def create_tables(self) -> None:
        """Create all database tables"""
        if not self.conn:
            raise RuntimeError("Not connected to database")

        for table_name in TABLES:
            sql = get_create_table_sql(table_name)
            # Convert MySQL syntax to SQLite
            sql = self._convert_mysql_to_sqlite(sql)
            try:
                self.cursor.execute(sql)
            except sqlite3.Error as e:
                logger.warning(f"Table {table_name} may already exist: {e}")

        self.conn.commit()
        logger.info(f"Created {len(TABLES)} tables")

    def _convert_mysql_to_sqlite(self, sql: str) -> str:
        """Convert MySQL SQL syntax to SQLite compatible syntax

        Args:
            sql: MySQL SQL statement

        Returns:
            SQLite compatible SQL statement
        """
        # Replace MySQL-specific types with SQLite equivalents
        replacements = {
            'AUTO_INCREMENT': 'AUTOINCREMENT',
            'INT ': 'INTEGER ',
            'INT,': 'INTEGER,',
            'INT)': 'INTEGER)',
            'BIGINT': 'INTEGER',
            'DECIMAL': 'REAL',
            'DOUBLE': 'REAL',
            'TIMESTAMP': 'DATETIME',
            'VARCHAR': 'TEXT',
            'CHAR': 'TEXT',
            'ENUM': 'TEXT',
            'JSON': 'TEXT',
            'ENGINE=InnoDB': '',
            'DEFAULT CHARSET=utf8mb4': '',
            'COMMENT=': '-- COMMENT=',
            '`': '"',  # MySQL backticks to SQLite double quotes
        }

        for mysql, sqlite in replacements.items():
            sql = sql.replace(mysql, sqlite)

        # Remove COMMENT clauses (SQLite doesn't support them)
        import re
        sql = re.sub(r'COMMENT\s*\'[^\']*\'', '', sql)

        return sql

    def execute(self, sql: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """Execute SQL statement

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Cursor object
        """
        if not self.cursor:
            raise RuntimeError("Not connected to database")

        # Convert MySQL %s placeholders to SQLite ?
        sql = sql.replace('%s', '?')

        if params:
            return self.cursor.execute(sql, params)
        else:
            return self.cursor.execute(sql)

    def fetch_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch single row

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Row as dictionary or None
        """
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        cursor = self.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert single row

        Args:
            table: Table name
            data: Row data as dictionary

        Returns:
            ID of inserted row
        """
        if not self.cursor:
            raise RuntimeError("Not connected to database")

        columns = ', '.join(f'"{k}"' for k in data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})'

        self.cursor.execute(sql, tuple(data.values()))
        return self.cursor.lastrowid

    def insert_many(self, table: str, data_list: List[Dict[str, Any]]) -> int:
        """Insert multiple rows

        Args:
            table: Table name
            data_list: List of row dictionaries

        Returns:
            Number of rows inserted
        """
        if not data_list:
            return 0

        if not self.cursor:
            raise RuntimeError("Not connected to database")

        columns = ', '.join(f'"{k}"' for k in data_list[0].keys())
        placeholders = ', '.join(['?' for _ in data_list[0]])
        sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})'

        values = [tuple(d.values()) for d in data_list]
        self.cursor.executemany(sql, values)
        return self.cursor.rowcount

    def update(self, table: str, data: Dict[str, Any],
               where: str, where_params: Tuple) -> int:
        """Update rows

        Args:
            table: Table name
            data: Update data
            where: WHERE clause
            where_params: WHERE parameters

        Returns:
            Number of rows updated
        """
        if not self.cursor:
            raise RuntimeError("Not connected to database")

        set_clause = ', '.join(f'"{k}" = ?' for k in data.keys())
        sql = f'UPDATE "{table}" SET {set_clause} WHERE {where}'

        params = tuple(data.values()) + where_params
        self.cursor.execute(sql, params)
        return self.cursor.rowcount

    def delete(self, table: str, where: str, where_params: Tuple) -> int:
        """Delete rows

        Args:
            table: Table name
            where: WHERE clause
            where_params: WHERE parameters

        Returns:
            Number of rows deleted
        """
        if not self.cursor:
            raise RuntimeError("Not connected to database")

        sql = f'DELETE FROM "{table}" WHERE {where}'
        self.cursor.execute(sql, where_params)
        return self.cursor.rowcount

    def begin_transaction(self) -> None:
        """Begin transaction (only if not already in transaction)"""
        # SQLite with autocommit disabled already handles transactions implicitly
        pass

    def commit(self) -> None:
        """Commit transaction"""
        if self.conn:
            self.conn.commit()

    def rollback(self) -> None:
        """Rollback transaction"""
        if self.conn:
            self.conn.rollback()

    def get_tables(self) -> List[str]:
        """Get list of all tables in database

        Returns:
            List of table names
        """
        cursor = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row['name'] for row in cursor.fetchall()]

    def table_exists(self, table: str) -> bool:
        """Check if table exists

        Args:
            table: Table name

        Returns:
            True if table exists
        """
        result = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        return result is not None
