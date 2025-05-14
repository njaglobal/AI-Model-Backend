from fastapi import FastAPI, UploadFile, File
from train import train_model
from predict import classify_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Model API is live"}

@app.post("/train")
def train_endpoint():
    success = train_model()
    return {"success": success}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    label = classify_image(contents)
    return {"prediction": label}
