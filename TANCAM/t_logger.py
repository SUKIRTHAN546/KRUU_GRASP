from logic.logger import log_violation

log_violation(
    camera_id="TEST_CAM",
    violations=["TEST_VIOLATION"],
    severity="TEST"
)

print("DONE")
