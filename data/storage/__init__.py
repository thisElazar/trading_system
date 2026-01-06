"""
Storage Module
==============
Database management for trades, research, and performance data.
"""

from data.storage.db_manager import DatabaseManager, get_db

__all__ = ['DatabaseManager', 'get_db']
