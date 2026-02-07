import pandas as pd
import streamlit as st
import cv2
import time
from main import process_frame

st.set_page_config(layout="wide")
st.title("🚧 Construction Safety Monitoring Dashboard")
col1, col2 = st.columns([2,1])
with col1:
    video_placeholder = st.empty()
with col2:
    st.subheader("Alerts & Violations")
    alert_placeholder = st.empty()
    violation_placeholder=st.empty()

# ---- INIT VIDEO ONCE ----
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture("KRUU_GRASP\\KRUU_GRASP\\videos\\test.mp4")  # or 0
st.caption("Green = Safe Zone|Red = High Risk Zone")
cap = st.session_state.cap

ret, frame = cap.read()
if not ret:
    st.warning("No frame received")
    st.stop()

frame, alert, all_violations = process_frame(frame)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
video_placeholder.image(frame_rgb, channels="RGB")
st.subheader("Alerts & Violations")
if alert != "INFO":
    rows = {}
    for sev, v, reason in all_violations:
        key = (v, reason)
        rows[key] = rows.get(key, 0) + 1

    data = []
    for (v, reason), count in rows.items():
        data.append({
            "Violation": v,
            "Reason": reason,
            "Count": count,
            "Severity": alert
        })

    df = pd.DataFrame(data)

    alert_placeholder.error(f"🚨 {alert}")
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    alert_placeholder.success("SAFE")
    st.dataframe(
        pd.DataFrame(columns=["Violation", "Reason", "Count", "Severity"]),
        use_container_width=True
    )

time.sleep(0.03)
st.rerun()

