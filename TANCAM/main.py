import cv2
import numpy as np
from matplotlib.pylab import rint
from ultralytics import YOLO
import argparse
from logic.logger import log_violation
from logic.perception import detect_ppe
from logic.abstraction import abstract_detection
from logic.context import is_person_at_height
from logic.rules import evaluate_ppe_rules
from logic.alerts import decide_alert_action
from logic.context import get_person_zone

violation_counter = 0
LAST_VIOLATION = None
parser = argparse.ArgumentParser(description="PPE Monitoring System")
parser.add_argument(
    "--mode",
    type=str,
    default="demo",
    choices=["demo", "video"],
    help="Run mode: demo (webcam) or video (file)"
)

parser.add_argument(
    "--video_path",
    type=str,
    default=None,
    help="Path to video file (used in video mode)"
)

args = parser.parse_args()
if args.mode == "demo":
    cap = cv2.VideoCapture(0)   
    camera_id = "CAM_DEMO"

elif args.mode == "video":
    if args.video_path is None:
        print("❌ Video path not provided")
        exit(1)
    cap = cv2.VideoCapture(args.video_path)
    camera_id = "CAM_VIDEO"
 
model = YOLO("TANCAM\\TANCAM\\model\\best.pt")
print("MODEL CLASSES:", model.names)
def build_contextual_reason(violation, zone, at_height):
    reasons = []

    if violation == "NO_HELMET":
        reasons.append("Helmet missing")

    if violation == "NO_HARNESS":
        reasons.append("Safety harness missing")

    if zone == "HIGH_RISK":
        reasons.append("in HIGH-RISK zone")

    if at_height:
        reasons.append("while working at height")

    return " ".join(reasons)



def process_frame(frame):
    h, w, _ = frame.shape
    all_violations = []

    # ---- ZONES ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (int(0.6*w), h), (0,255,0), -1)
    cv2.rectangle(overlay, (int(0.6*w), 0), (w, h), (0,0,255), -1)
    alpha = 0.15
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(overlay, "SAFE ZONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(overlay, "HIGH RISK ZONE", (int(0.6*w)+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ---- DETECTION ----
    persons = detect_ppe(frame, model)

    for p in persons:
        zone = get_person_zone(p, w)
        at_height = is_person_at_height(p["bbox"], h)
        violations = evaluate_ppe_rules(p, zone, at_height)
        for sev, v in violations:
            reason = build_contextual_reason(v, zone, at_height)
            all_violations.append((sev, v, reason))


    # ---- ALERT ----
    alert = decide_alert_action(all_violations)

    # ---- DRAW BOXES ----
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        color = (0,255,0)
        if alert == "WARNING":
            color = (0,255,255)
        elif alert == "CRITICAL":
            color = (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    # ---- ALERT TEXT ----
    if alert != "INFO":
        reasons = ", ".join([v[1] for v in all_violations])
        cv2.putText(frame, f"{alert}: {reasons}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    # ---- LEGEND ----
    cv2.putText(frame, "SAFE", (w-180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, "WARNING", (w-180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, "CRITICAL", (w-180, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame, alert, all_violations
if __name__ == "__main__":
    cap =cv2.VideoCapture("TANCAM\\TANCAM\\videos\\test.mp4")  # or "path/to/video.mp4"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, alert, all_violations = process_frame(frame)
    
        if alert != "INFO":
            if alert == LAST_VIOLATION:
                violation_counter += 1
            else:
                violation_counter = 1
                LAST_VIOLATION = alert
            if violation_counter >= 3:
                log_violation(
                camera_id=camera_id,
                violations=all_violations,
                severity=alert
                )
                print("✅ LOGGED:", alert, all_violations)
                violation_counter = 0
        if not all_violations:
            violation_counter = 0
            LAST_VIOLATION = None


        cv2.imshow("PPE Monitor", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()