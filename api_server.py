from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np

app = FastAPI(title="Smilo Emotion Detection API")

# Allow CORS for frontend usage 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
async def predict_emotion(file: UploadFile = File("image.png")):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][dominant_emotion]

        return JSONResponse(content={
            "emotion": dominant_emotion,
            "confidence": float(confidence)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
 
