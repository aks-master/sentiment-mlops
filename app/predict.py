from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and tokenizer
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Replace with your model path if local
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define input schema using Pydantic
class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_text: InputText):
    """
    Endpoint to make predictions on input text.
    """
    try:
        text = input_text.text

        # Check if text is provided
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Make prediction
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=-1).item()

        # Map sentiment to labels
        sentiment_label = "positive" if sentiment == 1 else "negative"

        return {
            "text": text,
            "sentiment": sentiment_label,
            "confidence": probs.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn (if running directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)