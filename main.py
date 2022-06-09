import uvicorn
from fastapi import FastAPI

from src.model_utils import get_index

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ping")
def read_root():
    return 'pong'


@app.get("/predict_index")
def predict_index(question: str):
    return get_index(question)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888)
