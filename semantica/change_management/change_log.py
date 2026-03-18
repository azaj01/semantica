"""
Change Log Module

This module provides standardized metadata structures for version changes
across both ontology and knowledge graph versioning systems.

Key Features:
    - Standardized ChangeLogEntry dataclass
    - Email validation for authors
    - Timestamp handling in ISO 8601 format
    - Optional change linking and tracking
    - Granular MutationRecord for per-entity audit trails

Main Classes:
    - ChangeLogEntry: Standard metadata for version changes
    - MutationRecord: Granular log of node/edge level changes

Example Usage:
    >>> from semantica.change_management.change_log import ChangeLogEntry
    >>> entry = ChangeLogEntry(
    ...     timestamp="2024-01-15T10:30:00Z",
    ...     author="alice@company.com",
    ...     description="Added Customer entity"
    ... )

Author: Semantica Contributors
License: MIT
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..utils.exceptions import ValidationError


@dataclass
class ChangeLogEntry:
    """
    Standard metadata for version changes.
    
    This dataclass provides a consistent structure for tracking changes
    across both ontology and knowledge graph versioning systems.
    
    Attributes:
        timestamp: ISO 8601 timestamp of the change
        author: Email address of the change author
        description: Description of the change (max 500 characters)
        change_id: Optional unique identifier for the change
        related_changes: Optional list of related change IDs
    """
    
    timestamp: str
    author: str
    description: str
    change_id: Optional[str] = None
    related_changes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self._validate_timestamp()
        self._validate_author()
        self._validate_description()
    
    def _validate_timestamp(self):
        """Validate timestamp is in ISO 8601 format."""
        try:
            # More strict validation for ISO 8601 format
            if 'T' not in self.timestamp:
                raise ValueError("Missing 'T' separator")
            datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise ValidationError(f"Invalid timestamp format: {self.timestamp}. Expected ISO 8601 format.")
    
    def _validate_author(self):
        """Validate author is a valid email address."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, self.author):
            raise ValidationError(f"Invalid email format: {self.author}")
    
    def _validate_description(self):
        """Validate description length."""
        if len(self.description) > 500:
            raise ValidationError(f"Description too long: {len(self.description)} characters (max 500)")
        if not self.description.strip():
            raise ValidationError("Description cannot be empty")
    
    @classmethod
    def create_now(cls, author: str, description: str, change_id: Optional[str] = None, 
                   related_changes: Optional[List[str]] = None) -> 'ChangeLogEntry':
        """
        Create a ChangeLogEntry with current timestamp.
        
        Args:
            author: Email address of the change author
            description: Description of the change
            change_id: Optional unique identifier for the change
            related_changes: Optional list of related change IDs
            
        Returns:
            ChangeLogEntry with current timestamp
        """
        return cls(
            timestamp=datetime.now().isoformat(),
            author=author,
            description=description,
            change_id=change_id,
            related_changes=related_changes or []
        )


@dataclass
class MutationRecord:
    """
    Granular record of a single change to a graph entity (node or edge).
    
    Attributes:
        timestamp: ISO 8601 timestamp of the mutation
        operation: 'ADD_NODE', 'UPDATE_NODE', 'REMOVE_NODE', 'ADD_EDGE', 'UPDATE_EDGE', 'REMOVE_EDGE'
        entity_id: The ID of the affected node or edge
        payload: The state of the entity after the mutation
        version_label: Optional association with a specific saved snapshot version
    """
    timestamp: str
    operation: str
    entity_id: str
    payload: Dict[str, Any]
    version_label: Optional[str] = None