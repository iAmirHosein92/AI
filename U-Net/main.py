from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from model import UNet  
from preprocess_image import preprocess_image
from postprocess import postprocess  


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("/kaggle/working/unet_chest_segmentation_v2.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    input_tensor = preprocess_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = (output > 0.5).float()

    # postprocess
    pred_mask = output.squeeze().cpu().numpy()
    clean_mask = postprocess(pred_mask)

    result_img = Image.fromarray(clean_mask.astype(np.uint8) * 255)
    buf = BytesIO()
    result_img.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type='image/png')