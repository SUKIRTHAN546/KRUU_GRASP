# abstraction.py

CLASS_MAP = {
    "Person": False,
    "Helmet": False,
    "Harness": False
}
def abstract_detection(results,model):
    flags = {
        "person": False,
        "helmet": False,
        "harness": False,
        "person_box": None
    }

    for box, cls in zip(results[0].boxes.xyxy,
                        results[0].boxes.cls):
        class_name = model.names[int(cls)]

        if class_name == "Person":
            flags["person"] = True
            flags["person_box"] = box

        elif class_name == "Helmet":
            flags["helmet"] = True

        elif class_name == "Harness":
            flags["harness"] = True

    return flags