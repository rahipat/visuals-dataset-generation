"""
Shared corrupt/empty-image handling for the *Dataset classes in data/ and
baselines/data/.

PIL raises OSError or UnidentifiedImageError on truncated, empty, or
otherwise malformed image files -- a known failure mode from partial writes
during dataset generation (e.g. disk-quota cutoffs mid-write). Rather than
let one bad file crash an entire training run, every dataset here retries
the next record (wrapping around the index) and logs what it skipped.
"""

import logging

from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)


def load_skipping_corrupt(n, idx, build_fn, context=""):
    """Call build_fn(i) for i starting at idx, advancing (wrapping) past any
    record whose image(s) raise OSError/UnidentifiedImageError. Returns
    (resolved_idx, build_fn's return value).

    build_fn must itself force image decode (Image.open(...).load() or
    .convert(...)) so corruption is caught here rather than downstream, and
    should raise rather than return partial data on failure.
    """
    prefix = f"[{context}] " if context else ""
    for offset in range(n):
        i = (idx + offset) % n
        try:
            return i, build_fn(i)
        except (OSError, UnidentifiedImageError) as e:
            logger.warning(
                "%sSkipping corrupt/unreadable sample at index %d: %s",
                prefix, i, e,
            )
    raise RuntimeError(f"{prefix}No readable samples found in dataset")
