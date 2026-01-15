from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import logging
from fastapi.middleware.cors import CORSMiddleware

#  Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Rate Limiter -------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment from text",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ------------------- CORS -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Load Sentiment Model -------------------
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    raise RuntimeError("Failed to load model or vectorizer: " + str(e))

# -------------------Rip Schema -------------------
class InputData(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text to analyze sentiment"
    )

# ------------------- Routes -------------------
@app.get("/")
def root():
    return {"message": "server is up and running"}

sentiment_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

#Rate limited endpoint
@app.post("/predict")
@limiter.limit("5/minute")  #limit
def predict_sentiment(request: Request, data: InputData):
    cleaned_text = data.text.strip().lower()

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error while processing the request")

    sentiment = sentiment_map.get(int(prediction))
    if sentiment is None:
        raise HTTPException(status_code=500, detail="Invalid model output")

    return {"prediction": sentiment}
