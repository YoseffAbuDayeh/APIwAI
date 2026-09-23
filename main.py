from fastapi import FastAPI
from controllers import *
from controllers import generalController

app = FastAPI()

app.include_router(generalController.router, tags=["general"])


#To get the API UI then one has to do /docs
@app.get("/")
async def root():
    return {"message": "Hello World"}

