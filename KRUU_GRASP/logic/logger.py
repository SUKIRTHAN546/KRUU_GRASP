import csv
import os
from datetime import datetime

LOG_FILE = "KRUU_GRASP\\KRUU_GRASP\\logs\\violations.csv"

def log_violation(camera_id, violations, severity):
    """
    Logs safety violations to CSV file
    """

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow([
                "timestamp",
                "camera_id",
                "violations",
                "severity"
            ])
        

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_id,
            ", ".join([v[1] for v in violations]),
            severity
        ])
