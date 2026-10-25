"""Evidence parsers.

Every parser turns one evidence file into a list of :class:`Observation`
objects.  An observation is one grounded fact about one resource at one point
in time: whether it exists, which direct identifiers it still carries, whether
a connector is enabled, whether a backup can be restored, and so on.  The
rest of the engine never looks at raw files again.
"""

from .base import Identifiers, Observation, parse_all_evidence  # noqa: F401

__all__ = ["Identifiers", "Observation", "parse_all_evidence"]
