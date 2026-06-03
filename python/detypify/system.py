"""System compatibility helpers."""

from os import cpu_count


def available_cpu_count() -> int:
    """Return the CPU count available to this process, respecting affinity when possible."""
    try:
        from os import sched_getaffinity

        return len(sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return cpu_count() or 1
