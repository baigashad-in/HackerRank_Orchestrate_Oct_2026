"""Ghost Data Still Exists: deletion-audit engine.

Package layout:
    config.py              constants (systems, operations, ordering)
    loaders.py             CSV loaders for the participant-facing dataset
    identity.py            current + historical identifier resolution
    request_classifier.py  deletion vs non-deletion request classification
    evidence/              parsers turning every evidence file into Observations
    activity.py            system_activity.csv indexing and deletion-worker batches
    linking.py             resource <-> user linking
    state.py               state reconstruction at audit_at
    planner.py             policy application + ordered step generation
    pipeline.py            end-to-end orchestration
    llm.py                 optional model layer (off unless an API key is set)
"""

__version__ = "1.0.0"
