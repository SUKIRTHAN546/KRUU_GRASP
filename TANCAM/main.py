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
 
model = YOLO("KRUU\\TANCAM\\TANCAM\\model\\best.pt")
print("MODEL CLASSES:", model.names)

if args.mode == "demo":
    cap = cv2.VideoCapture(0)   
    camera_id = "CAM_DEMO"

elif args.mode == "video":
    if args.video_path is None:
        print("❌ Video path not provided")
        exit(1)
    cap = cv2.VideoCapture(args.video_path)
    camera_id = "CAM_VIDEO"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons = detect_ppe(frame, model)

    flags = abstract_detection(persons)
    h,w,_=frame.shape
    

# SAFE zone (green)
    cv2.rectangle(frame, (0, 0), (int(0.6*w), h), (0,255,0), 2)

# HIGH RISK zone (red)
    cv2.rectangle(frame, (int(0.6*w), 0), (w, h), (0,0,255), 2)

    all_violations = []
    for p in persons:
        zone = get_person_zone(p,w)
        print("ZONE:", zone)
        at_height = is_person_at_height(p["bbox"], h)

        violations = evaluate_ppe_rules(p, zone, at_height)
        all_violations.extend(violations)
    
    alert = decide_alert_action(all_violations)

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]


        color = (0,255,0)  # default safe

        if alert == "WARNING":
                color = (0,255,255)  # yellow
        elif alert == "CRITICAL":
                color = (0,0,255)  # red

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    
    if alert != "INFO":
        reasons = ", ".join([v[1] for v in all_violations])

        text = f"{alert}: {reasons}"

        cv2.putText(
            frame,
            text,
            (20, 40),                  # top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),                 # red text
            3
        )

    
    print("----- FRAME DEBUG -----")
    print("FLAGS:", flags)
    print("ALL VIOLATIONS:", all_violations)
    print("ALERT:", alert)
    print("------------------------")
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


    print("PERSONS:", persons)
    print("FLAGS:", flags)

    if alert == "INFO":
        status_text = "SAFE"
        color = (0, 255, 0)
    elif alert == "WARNING":
        status_text = "PPE WARNING"
        color = (0, 255, 255)
    else:
        status_text = "CRITICAL ALERT"
        color = (0, 0, 255)

    
    cv2.imshow("PPE Monitor", frame)        
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    



cap.release()
cv2.destroyAllWindows()