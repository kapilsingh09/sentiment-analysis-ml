from fastapi import FastAPI , HTTPException
from pydantic import BaseModel, Field
import joblib
import logging
from fastapi.middleware.cors import CORSMiddleware

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", description="API for predicting sentiment from text", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    raise RuntimeError("Failed to load model or vectorizer: " + str(e))

class InputData(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text to analyze sentiment"
    )

@app.get("/")
def root():
    return {"message":"server is up and running"}



sentiment_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "suprise",
    4: "fear",
    5: "joy"
}

@app.post("/predict")
@app.post("/predict")
def predict_sentiment(data: InputData):
    cleaned_text = data.text.strip().lower()  

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error while processing the request"
        )

    sentiment = sentiment_map.get(int(prediction))
    if sentiment is None:
        raise HTTPException(status_code=500, detail="Invalid model output")

    return {"prediction": sentiment}


