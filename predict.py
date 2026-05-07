import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



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
        inputs = tokenizer(sentence,return_tensors="pt")
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1)
        print(prediction)
        
