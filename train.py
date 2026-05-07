from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

"""Loading the train split from the dataset sst2 (The Stanford Sentiment Treebank is a corpus with
    fully labeled parse trees that allows for a complete analysis of the compositional 
    effects of sentiment in language.stanford sentiment analysis from movie reviews)"""
ds = load_dataset("stanfordnlp/sst2")

# Loading in bert base uncased for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

# Loading in the corresponding tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Creating tokenization function
def tokenize(data):
    # Accessing the sentence column of the dataset
    return bert_tokenizer(data['sentence'], truncation=True,padding=True)

# Using the custom tokenize function to map the dataset
tokenized_ds = ds.map(tokenize, batched=True)

# Here I am creating my own training loop. However, this is just for practice

# Initializing the optimizer
optimizer = torch.optim.AdamW(model.parameters)

# Using data loader to batch the data, and setting shuffle to be on (to prevent the model from learning
# extraneous patterns)
loaded_data = torch.utils.data.DataLoader(tokenized_ds["train"], shuffle=True)

# Training loop
for epoch in range(3):
    for batch in loaded_data:
        # Resetting the gradients
        optimizer.zero_grad()
        # Getting the outputs after unpacking the batch dictionary
        outputs = model(**batch)
        # Calculating the loss on the outputs
        loss = outputs.loss
        # Backpropagating the loss
        loss.backwards()
        # Pushing the optimizer one step forward
        optimizer.step()

# Saving the model after training
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")


