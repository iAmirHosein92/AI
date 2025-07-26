from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

from model import load_model
from utils import preprocess_image, postprocess_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(device=device)

app = FastAPI()

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    image_tensor = preprocess_image(await file.read())
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        pred = model(image_tensor)
    
    mask = postprocess_mask(pred)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")