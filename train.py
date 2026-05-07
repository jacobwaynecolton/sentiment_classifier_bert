from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import DataCollatorWithPadding
from tqdm import tqdm

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
    return bert_tokenizer(data['sentence'], truncation=True)

# Using the custom tokenize function to map the dataset
tokenized_ds = ds.map(tokenize, batched=True)
# Removing unnecessary columns
tokenized_ds = tokenized_ds.remove_columns(["sentence", "idx"])
# Renaming the label column to labels
tokenized_ds = tokenized_ds.rename_column("label", "labels")
# Formatting the dataset for pytorch
tokenized_ds.set_format("torch")

# Here I am creating my own training loop. However, this is just for practice

# Initializing the optimizer
optimizer = torch.optim.AdamW(model.parameters())

# Setting up a collator
collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Using data loader to batch the data, and setting shuffle to be on (to prevent the model from learning
# extraneous patterns)
loaded_data = torch.utils.data.DataLoader(tokenized_ds["train"], batch_size = 16,shuffle=True, collate_fn = collator)

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model.to(device)

# Training loop
for epoch in range(3):
    print(f"Epoch {epoch+1}/3")
    for batch in tqdm(loaded_data):
        # Moving batch items to gpu
        batch = {k: v.to(device) for k, v in batch.items()}
        # Resetting the gradients
        optimizer.zero_grad()
        # Getting the outputs after unpacking the batch dictionary
        outputs = model(**batch)
        # Calculating the loss on the outputs
        loss = outputs.loss
        # Backpropagating the loss
        loss.backward()
        # Pushing the optimizer one step forward
        optimizer.step()

# Saving the model after training
model.save_pretrained("./model")
bert_tokenizer.save_pretrained("./model")


