

def is_person_at_height(person_box, image_height, threshold=0.8):
    """
    Determines if a person is working at height
    based on bounding box vertical position
    """
    if person_box is None:
        return False
    y_bottom = person_box[3]
    if y_bottom / image_height < threshold:
        return True
    else:
        return False