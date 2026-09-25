from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image as PILImage

import io
import torch

from methods import imageHelper, ollamaPart

router = APIRouter(
    prefix="/general",
    tags=["general"],
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@router.get("/hasCuda")
def hasCuda():
    '''
    API call to check if cuda is available

    :return: True or False depending on if cuda is available
    '''

    #I use an AMD card so I can't run it with cuda cores
    return JSONResponse(status_code=200, content={"Cuda": torch.cuda.is_available()})

@router.post("/image")
async def image(file: UploadFile = File(...)):
    '''
    API call to obtain contents from an image file and output them as a JSON

    :param file:        Image file to scan and get the data from
    :return:            JSON with data from the image
    '''
    allowed_extensions = {"image/png", "image/jpg", "image/jpeg"}
    #If file is not a png, jpeg, or jpg then it will exit with a 400 error code.
    if file.content_type not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported Media Type")


    contents = await file.read()
    #Reads the file and converts it into RGB. This will normalize the image in-case it has alpha channels
    image = PILImage.open(io.BytesIO(contents)).convert("RGB")

    #Calls the method and stores the lines in line_images
    line_images = imageHelper.segmentLines(image)
    print(f"Number of lines detected: {len(line_images)}")

    print(line_images)

    extracted_lines = []
    #It uses the TrOCRP pretrained model to read the data from the image
    for idx, line_img in enumerate(line_images):
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=200, num_beams=5)
        #max_new_tokens is the max number of words that it will show.
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Line {idx}: {text}")
        extracted_lines.append(text)

    extracted_text = "\n".join(extracted_lines)


    print("Extracted text = " + extracted_text)

    # Makes the extracted text a JSON using the Local LLM
    extracted_text = ollamaPart.json_maker(extracted_text)

    print(extracted_text)
    return {"extracted_text": extracted_text}
