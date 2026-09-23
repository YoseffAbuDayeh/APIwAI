from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/general",
    tags=["general"],
)

@router.get("/helloWorld")
def hello_world():
    return JSONResponse(status_code=200, content={"message": "Hello World"})