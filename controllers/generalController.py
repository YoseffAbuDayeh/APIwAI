from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image as PILImage

import io
import torch

from methods import imageHelper

router = APIRouter(
    prefix="/general",
    tags=["general"],
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@router.get("/helloWorld")
def hello_world():
    print("has cuda: ", torch.cuda.is_available())
    return JSONResponse(status_code=200, content={"message": "Hello World"})

@router.post("/image")
async def image(file: UploadFile = File(...)):
    contents = await file.read()
    image = PILImage.open(io.BytesIO(contents)).convert("RGB")

    line_images = imageHelper.segmentLines(image)
    print(f"Number of lines detected: {len(line_images)}")

    extracted_lines = []
    for idx, line_img in enumerate(line_images):
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=200, num_beams=5)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Line {idx}: {text}")
        extracted_lines.append(text)

    extracted_text = "\n".join(extracted_lines)


    print("Extracted text = " + extracted_text)

    return {"extracted_text": extracted_text}
