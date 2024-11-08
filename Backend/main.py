from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ok"}

@app.post("/classify")
async def classify_dog_breed(file: UploadFile = File(...)):
    # Placeholder
    return {"filename": file.filename, "message": "Dog breed classification in development"}
