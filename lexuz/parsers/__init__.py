"""
lexuz.parsers
"""

from lexuz.parsers.archetype_a import ArchetypeAParser, parse_structural_header
from lexuz.parsers.table_parser import (
    format_table_markdown,
    parse_table_grid,
    table_to_markdown,
    table_to_markdown_slices,
)

__all__ = [
    "ArchetypeAParser",
    "parse_structural_header",
    "parse_table_grid",
    "format_table_markdown",
    "table_to_markdown",
    "table_to_markdown_slices",
]

