from fastapi import FastAPI
from controllers import *
from controllers import generalController

app = FastAPI()

app.include_router(generalController.router, tags=["general"])

@app.get("/")
async def root():
    return {"message": "Hello World"}

