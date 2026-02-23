"""
数据库模式基础定义

包含枚举类型、列定义、索引定义和表定义基类
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any


class ColumnType(Enum):
    """数据库列类型"""
    VARCHAR = "varchar"
    CHAR = "char"
    INT = "int"
    BIGINT = "bigint"
    SMALLINT = "smallint"
    DECIMAL = "decimal"
    FLOAT = "float"
    DOUBLE = "double"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    DATE = "date"
    TIME = "time"
    BOOLEAN = "boolean"
    TEXT = "text"
    BLOB = "blob"
    JSON = "json"
    ENUM = "enum"


class ConstraintType(Enum):
    """约束类型"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    INDEX = "index"


class TableCategory(Enum):
    """表分类"""
    CONFIG = "config"           # 配置层
    SCENARIO = "scenario"       # 场景实例层
    DECOMPOSITION = "decomposition"  # 目标分解层
    EXPERIMENT = "experiment"   # 实验层
    RESULT = "result"           # 结果层
    STATE = "state"             # 状态层
    NETWORK = "network"         # 网络层


@dataclass
class ColumnDefinition:
    """列定义"""
    name: str
    col_type: ColumnType
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default: Any = None
    auto_increment: bool = False
    comment: str = ""
    enum_values: Optional[List[str]] = None

    def to_sql(self) -> str:
        """生成列定义的SQL"""
        type_str = self.col_type.value.upper()

        if self.length and self.col_type in (ColumnType.VARCHAR, ColumnType.CHAR):
            type_str += f"({self.length})"
        elif self.precision is not None and self.scale is not None:
            type_str += f"({self.precision},{self.scale})"
        elif self.length and self.col_type == ColumnType.DECIMAL:
            type_str += f"({self.length},{self.scale or 2})"
        elif self.enum_values and self.col_type == ColumnType.ENUM:
            values = ",".join(f"'{v}'" for v in self.enum_values)
            type_str += f"({values})"

        parts = [f"`{self.name}`", type_str]

        if not self.nullable:
            parts.append("NOT NULL")

        if self.default is not None:
            if isinstance(self.default, str):
                parts.append(f"DEFAULT '{self.default}'")
            else:
                parts.append(f"DEFAULT {self.default}")

        if self.auto_increment:
            parts.append("AUTO_INCREMENT")

        return " ".join(parts)


@dataclass
class IndexDefinition:
    """索引/约束定义"""
    name: str
    constraint_type: ConstraintType
    columns: List[str]
    reference_table: Optional[str] = None
    reference_columns: Optional[List[str]] = None
    on_delete: Optional[str] = None

    def to_sql(self) -> str:
        """生成约束SQL"""
        if self.constraint_type == ConstraintType.PRIMARY_KEY:
            cols = ",".join(f"`{c}`" for c in self.columns)
            return f"PRIMARY KEY ({cols})"
        elif self.constraint_type == ConstraintType.UNIQUE:
            cols = ",".join(f"`{c}`" for c in self.columns)
            return f"UNIQUE KEY `{self.name}` ({cols})"
        elif self.constraint_type == ConstraintType.FOREIGN_KEY:
            cols = ",".join(f"`{c}`" for c in self.columns)
            ref_cols = ",".join(f"`{c}`" for c in (self.reference_columns or []))
            sql = f"FOREIGN KEY ({cols}) REFERENCES `{self.reference_table}` ({ref_cols})"
            if self.on_delete:
                sql += f" ON DELETE {self.on_delete}"
            return sql
        elif self.constraint_type == ConstraintType.INDEX:
            cols = ",".join(f"`{c}`" for c in self.columns)
            return f"INDEX `{self.name}` ({cols})"
        return ""


@dataclass
class TableDefinition:
    """表定义基类"""
    name: str
    category: TableCategory
    columns: List[ColumnDefinition]
    constraints: List[IndexDefinition] = field(default_factory=list)
    comment: str = ""
    engine: str = "InnoDB"
    charset: str = "utf8mb4"

    def to_sql(self) -> str:
        """生成建表SQL"""
        lines = [f"CREATE TABLE `{self.name}` ("]

        # 列定义
        col_lines = ["    " + col.to_sql() for col in self.columns]

        # 约束定义
        constraint_lines = ["    " + c.to_sql() for c in self.constraints if c.constraint_type != ConstraintType.INDEX]

        # 索引定义
        index_lines = ["    " + c.to_sql() for c in self.constraints if c.constraint_type == ConstraintType.INDEX]

        all_lines = col_lines + constraint_lines + index_lines
        lines.append(",\n".join(all_lines))
        lines.append(")")

        lines.append(f"ENGINE={self.engine}")
        lines.append(f"DEFAULT CHARSET={self.charset}")
        if self.comment:
            lines.append(f"COMMENT='{self.comment}'")

        return "\n".join(lines)
