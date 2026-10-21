"""
Indexing-time privacy: detect personal information + secrets, owner review,
decisions, redaction and verification (plans/privacy-pii-redaction-plan-v3.md).

Presidio / spaCy / detect-secrets are imported lazily inside functions, and
this package is never imported by hypeonoath.serving (boundary test). The
shared matcher every side uses lives in hypeonoath.core.private_values.
"""
