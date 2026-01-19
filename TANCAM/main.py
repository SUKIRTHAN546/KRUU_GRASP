import cv2
from matplotlib.pylab import rint
from ultralytics import YOLO
import argparse
from logic.logger import log_violation

from logic.abstraction import abstract_detection
from logic.context import is_person_at_height
from logic.rules import evaluate_ppe_rules
from logic.alerts import decide_alert_action

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
 
model = YOLO("model/best.pt")

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

    results = model(frame, conf=0.15)

    flags = abstract_detection(results, model)
    image_height = frame.shape[0]
    at_height = is_person_at_height(flags["person_box"], image_height)

    violations = evaluate_ppe_rules(flags, at_height)
    alert = decide_alert_action(violations)
    print("----- FRAME DEBUG -----")
    print("FLAGS:", flags)
    print("AT_HEIGHT:", at_height)
    print("VIOLATIONS:", violations)
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
            violations=violations,
            severity=alert
            )
            print("✅ LOGGED:", alert, violations)
            violation_counter = 0
    else:
        violation_counter = 0
        LAST_VIOLATION = None



    if alert == "INFO":
        status_text = "SAFE"
        color = (0, 255, 0)
    elif alert == "WARNING":
        status_text = "PPE WARNING"
        color = (0, 255, 255)
    else:
        status_text = "CRITICAL ALERT"
        color = (0, 0, 255)

    annotated_frame = results[0].plot()
    cv2.putText(
        annotated_frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("PPE Monitoring System", annotated_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    



cap.release()
cv2.destroyAllWindows()