from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from train import train_model
from predict import classify_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    result = classify_image(contents)
    return result  # 🔥 Return the full dict directly
