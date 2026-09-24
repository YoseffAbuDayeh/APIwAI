from fastapi import FastAPI
from controllers import *
from controllers import generalController

app = FastAPI()

#We need to add one of these for each controller file we have. Basically we are telling it (We have controllers in this file)
app.include_router(generalController.router, tags=["general"])


#To get the API UI then one has to do /docs
@app.get("/")
async def root():
    return {"message": "Hello World"}

