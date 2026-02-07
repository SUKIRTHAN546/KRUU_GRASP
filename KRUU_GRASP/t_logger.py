from ultralytics import YOLO

model = YOLO("C:\\Users\\sukir\\OneDrive\\Documents\\TANCAM PROJECT\\KRUU\\TANCAM\\TANCAM\\model\\best.pt")

print(model.names)
