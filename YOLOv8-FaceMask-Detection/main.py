from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import shutil, uuid, os

app = FastAPI()

# Load the trained YOLOv8 model
model = YOLO("runs/detect/mask_detector_v2/weights/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    output_dir = os.path.join(UPLOAD_DIR, "results")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(input_path, save=True, project=output_dir, name=image_id)
    output_path = os.path.join(output_dir, image_id, f"{image_id}.jpg")

    return FileResponse(output_path, media_type="image/jpeg")

@app.post("/predict-json/")
async def predict_json(file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(input_path)

    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf = round(float(box.conf[0]), 2)
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            predictions.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy
            })

    return JSONResponse(content=predictions)