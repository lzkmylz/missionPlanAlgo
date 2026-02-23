"""Repository pattern for entity storage."""
from .base import EntityRepository
from .json_repository import JSONEntityRepository

__all__ = ['EntityRepository', 'JSONEntityRepository']
