

def detect_ppe(frame, model):
    

    results = model(frame, conf=0.15)

    persons = []

    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        
        harness = cls == 0
        helmet = cls == 1
        person = cls == 2        

        persons.append({
            "person_id": i,
            "helmet": helmet,
            "harness": harness,
            "bbox": (x1, y1, x2, y2)
        })

    return persons
