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


def load_skipping_corrupt(n, idx, build_fn, context="", bad_indices=None):
    """Call build_fn(i) for i starting at idx, advancing (wrapping) past any
    record whose image(s) raise OSError/UnidentifiedImageError. Returns
    (resolved_idx, build_fn's return value).

    build_fn must itself force image decode (Image.open(...).load() or
    .convert(...)) so corruption is caught here rather than downstream, and
    should raise rather than return partial data on failure.

    bad_indices, if given, is a set the caller owns (typically one per
    Dataset instance, e.g. self._bad_indices set up in __init__). A
    permanently-corrupt file is hit again every epoch under the default
    without-replacement sampler -- that's expected, not a retry-loop bug --
    but there's no reason to re-open and re-decode a file we already know is
    bad, or to re-log it every time. Once an index is confirmed bad it's
    skipped outright on later calls (no build_fn attempt, no new log line)
    for the remaining lifetime of whatever owns the set. Note this is
    per-process: with num_workers>0 each DataLoader worker holds its own
    copy, so a file can still log up to num_workers times total (once per
    worker's first encounter) rather than exactly once dataset-wide.
    """
    prefix = f"[{context}] " if context else ""
    for offset in range(n):
        i = (idx + offset) % n
        if bad_indices is not None and i in bad_indices:
            continue
        try:
            return i, build_fn(i)
        except (OSError, UnidentifiedImageError) as e:
            if bad_indices is not None:
                bad_indices.add(i)
            logger.warning(
                "%sSkipping corrupt/unreadable sample at index %d: %s",
                prefix, i, e,
            )
    raise RuntimeError(f"{prefix}No readable samples found in dataset")
