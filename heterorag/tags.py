"""
heterorag/tags.py
=================
Tag-string parsing shared by the loader, the repair script and the result
normaliser.

Stack Exchange dumps have used two encodings for the Posts.xml ``Tags``
attribute:

    legacy : "<python><pandas>"   (attribute value after XML unescaping)
    current: "|python|pandas|"    (dumps from late 2023 onward)

Code that assumed only the legacy form silently produced one garbage tag
("|python|pandas|") per question on current dumps. Everything that needs to
turn a raw tag string into names should go through ``parse_tags``.
"""

from __future__ import annotations

import re

_ANGLE = re.compile(r"<([^<>]+)>")


def parse_tags(raw: str | None) -> list[str]:
    """Return tag names from either dump encoding, in order, without blanks."""
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if "|" in raw:
        return [t.strip() for t in raw.split("|") if t.strip()]
    found = _ANGLE.findall(raw)
    if found:
        return [t.strip() for t in found if t.strip()]
    # Plain comma-separated fallback (used by some hand-written fixtures).
    return [t.strip() for t in raw.split(",") if t.strip()]
