from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Creating the FastAPI app
app = FastAPI()

# Setting the path to the saved model folder
model_path = "./model"

# Loading in the fine-tuned model from the local model folder
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Loading in the corresponding tokenizer from the local model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Switching the model into evaluation mode
model.eval()
# Creating a dictionary to map the model output numbers to readable labels
labels = {
    0: "negative",
    1: "positive"
}
# Creating the expected format for the incoming request
# The request should contain a text field with the sentence to classify
class PredictionRequest(BaseModel):
    text: str


# Creating prediction function for sentiment analysis
def predict_sentiment(text: str):
    # Tokenizing the input sentence and converting it into pytorch tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    # Turning off computational graph generation
    with torch.no_grad():
        # Getting the model outputs after unpacking the input dictionary
        outputs = model(**inputs)

    # Applying softmax to convert the raw logits into probabilities
    probs = F.softmax(outputs.logits, dim=-1)

    # Getting the index of the highest probability
    prediction = probs.argmax(dim=-1).item()

    # Getting the confidence score associated with the prediction
    confidence = probs[0][prediction].item()

    # Returning the readable label and confidence score
    return {
        "label": labels[prediction],
        "confidence": round(confidence, 4)
    }

# Creating a basic root endpoint to check that the API is running
@app.get("/")
def root():
    return {
        "message": "Sentiment classifier API is running."
    }

# Creating the prediction endpoint
# This receives a POST request containing text and returns the sentiment prediction
@app.post("/predict")
def predict(request: PredictionRequest):
    # Running the prediction function on the text from the request
    result = predict_sentiment(request.text)

    # Returning the result as JSON
    return result