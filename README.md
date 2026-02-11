# 🎭 Sentiment Analysis API

A machine learning-powered REST API that analyzes text and predicts emotions using Natural Language Processing (NLP). Built with FastAPI and scikit-learn, this API can classify text into six different emotional categories.

## 📊 Features

- **Multi-class Emotion Detection**: Classifies text into 6 emotions:
  - 😢 Sadness
  - 😠 Anger
  - ❤️ Love
  - 😲 Surprise
  - 😨 Fear
  - 😊 Joy

- **FastAPI Framework**: High-performance async API with automatic documentation
- **Rate Limiting**: Built-in protection against API abuse (5 requests/minute)
- **CORS Enabled**: Ready for cross-origin requests
- **Input Validation**: Pydantic models for request validation
- **TF-IDF Vectorization**: Advanced text feature extraction
- **Logistic Regression Model**: Trained on comprehensive emotion dataset

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nlp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requriment.txt
   ```

3. **Additional NLP resources** (for model training)
   ```bash
   pip install nltk pandas numpy seaborn matplotlib
   ```

### Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Endpoints

#### `GET /`
Health check endpoint to verify server status.

**Response:**
```json
{
  "message": "server is up and running"
}
```

#### `POST /predict`
Predict the emotion/sentiment of provided text.

**Rate Limit:** 5 requests per minute per IP

**Request Body:**
```json
{
  "text": "I am so happy today!"
}
```

**Response:**
```json
{
  "prediction": "joy"
}
```

**Constraints:**
- Text length: 1-1000 characters
- Cannot be empty or only whitespace

### Interactive Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧠 Model Training

The sentiment analysis model is trained using the following pipeline:

### Data Preprocessing
1. Lowercase conversion
2. Punctuation removal
3. Number removal
4. Emoji and non-ASCII character removal
5. Stopword removal (NLTK)

### Feature Extraction
- **TF-IDF Vectorization**: Converts text into numerical features

### Model
- **Algorithm**: Logistic Regression
- **Training**: 80/20 train-test split
- **Iterations**: 10,000 max iterations

### Training the Model

Run the training script:

```bash
python main_nlp.py
```

This will:
- Load and preprocess the training data (`train.txt`)
- Train the model
- Save `sentiment_model.pkl` and `tfidf_vectorizer.pkl`

## 📁 Project Structure

```
nlp/
├── main.py                  # FastAPI application
├── main_nlp.py             # Model training script
├── sentiment_model.pkl     # Trained model (generated)
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer (generated)
├── train.txt               # Training dataset
├── test.txt                # Test dataset
├── requriment.txt          # Dependencies
├── npl.ipynb               # Jupyter notebook (exploration)
└── README.md               # This file
```

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **Pydantic**: Data validation using Python type annotations

### Machine Learning
- **scikit-learn**: Machine learning library (Logistic Regression, TF-IDF)
- **NLTK**: Natural Language Toolkit for text preprocessing
- **joblib**: Model serialization

### Security & Performance
- **SlowAPI**: Rate limiting middleware
- **CORS Middleware**: Cross-origin resource sharing

## 🔒 Security Features

- **Rate Limiting**: 5 requests per minute per IP address
- **Input Validation**: Text length limits (1-1000 characters)
- **Error Handling**: Comprehensive exception handling
- **Logging**: Request and error logging for monitoring

## 📝 Example Usage

### Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
data = {"text": "I love this amazing project!"}

response = requests.post(url, json=data)
print(response.json())
# Output: {"prediction": "love"}
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling wonderful today!"}'
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'This is absolutely amazing!'})
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🎯 Emotion Label Mapping

| Code | Emotion  | Example Text                    |
|------|----------|---------------------------------|
| 0    | Sadness  | "I feel so alone and lost"      |
| 1    | Anger    | "This is completely unacceptable!" |
| 2    | Love     | "I adore spending time with you" |
| 3    | Surprise | "Wow, I never expected this!"   |
| 4    | Fear     | "I'm really worried about this" |
| 5    | Joy      | "I'm so happy and excited!"     |

## 🐛 Troubleshooting

### Model files not found
Make sure to run `main_nlp.py` first to generate the model files:
```bash
python main_nlp.py
```

### NLTK data not found
Download required NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Rate limit exceeded
Wait 1 minute before making additional requests, or adjust the rate limit in `main.py`:
```python
@limiter.limit("5/minute")  # Change to your desired limit
```

## 📈 Performance

- **Model Accuracy**: Check console output after training
- **Response Time**: ~50-100ms per prediction
- **Concurrent Requests**: Supports async processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ using FastAPI and scikit-learn

---

**Note**: This is a machine learning project for educational purposes. The model's predictions may not always be 100% accurate and should be used accordingly.
