import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F



# Importing the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Switching the model into evaluation mode
model.eval()

while True:
    sentence = input("Please give me your example sentence for sentiment analysis: \n")
    if sentence.lower() == "end":
        break

    # Turning off computational graph generation
    with torch.no_grad():
        print(sentence)
        inputs = tokenizer(sentence,return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = probs.argmax(dim=-1).item()
        confidence = probs[0][prediction].item()
    
        labels = {0: "negative", 1: "positive"}
        print(f"{labels[prediction]} ({confidence:.0%} confidence)")
