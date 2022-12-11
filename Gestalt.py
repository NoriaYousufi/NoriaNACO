import difflib

# Dqr Implementation in Python
def quick_ratio(s1: str, s2: str) -> float:
    """Return an upper bound on ratio() relatively quickly."""
    length = len(s1) + len(s2)

    if not length:
        return 1.0

    intersect = collections.Counter(s1) & collections.Counter(s2)
    matches = sum(intersect.values())
    return 2.0 * matches / length