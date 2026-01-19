

INFO = "INFO"
WARNING = "WARNING"
CRITICAL_ALERT = "CRITICAL_ALERT"

def decide_alert_action(violations):
    """
    Decides alert severity and action based on violations
    """

    if "CRITICAL_VIOLATION" in violations:
        return CRITICAL_ALERT

    if "NO_HELMET" in violations or "NO_HARNESS" in violations:
        return WARNING

    
    return INFO

