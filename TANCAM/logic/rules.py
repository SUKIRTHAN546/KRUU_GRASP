# rules.py

SAFE = "SAFE"
NO_HELMET = "NO_HELMET"
NO_HARNESS = "NO_HARNESS"
CRITICAL = "CRITICAL_VIOLATION"
def evaluate_ppe_rules(flags, at_height):
    """
    Applies PPE safety rules and returns violation status
    """

    violations = []

    # Rule 1: Helmet rule (always applicable)
    if flags["person"] and not flags["helmet"]:
        violations.append(NO_HELMET)

    # Rule 2: Harness rule (only at height)
    if flags["person"] and at_height and not flags["harness"]:
        violations.append(NO_HARNESS)

    # Rule 3: Critical case
    if flags["person"] and at_height and (NO_HARNESS in violations or NO_HELMET in violations):
        violations.append(CRITICAL)

    if not violations:
        return [SAFE]

    return violations
