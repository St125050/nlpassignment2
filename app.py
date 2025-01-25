import requests
import nltk
import re
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset (example for Project Gutenberg text)
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
data = response.text

# Save the data to a file
with open("dataset.txt", "w", encoding="utf-8") as file:
    file.write(data)

# Load the dataset from file
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
tokens = nltk.word_tokenize(text)

# Lowercasing
tokens = [token.lower() for token in tokens]

# Removing punctuation and special characters
tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]

# Removing stop words (optional)
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words]

# Add a special token for unknown words
tokens.append('<UNK>')

# Numericalization
vocab = list(set(tokens))
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for i, word in enumerate(vocab)}

# Creating sequences
sequence_length = 5
sequences = []
for i in range(len(tokens) - sequence_length):
    sequences.append(tokens[i:i + sequence_length])

# Convert sequences to numerical indices
input_sequences = []
for sequence in sequences:
    input_sequences.append([word2index[word] for word in sequence])

# Convert to numpy array
input_sequences = np.array(input_sequences)

# Define the Language Model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        x = self.embedding(x)
        x, state = self.lstm(x, prev_state)
        x = self.fc(x)
        return x, state

    def init_state(self, batch_size=1):
        return (torch.zeros(2, batch_size, self.lstm.hidden_size),
                torch.zeros(2, batch_size, self.lstm.hidden_size))

# Hyperparameters
embedding_dim = 50
hidden_dim = 100
vocab_size = len(vocab)
batch_size = 32
epochs = 10

# Model, loss function, optimizer
model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, input_sequences, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(input_sequences) - batch_size, batch_size):
            inputs = torch.tensor(input_sequences[i:i + batch_size, :-1], dtype=torch.long)
            targets = torch.tensor(input_sequences[i:i + batch_size, 1:], dtype=torch.long)

            optimizer.zero_grad()
            state_h, state_c = model.init_state(batch_size)
            outputs, _ = model(inputs, (state_h, state_c))
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(input_sequences) // batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Train the model
train_model(model, input_sequences, criterion, optimizer, epochs)

# Save the model's state dictionary
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully.")

# Text generation function
def generate_text(model, start_text, max_length=50):
    model.eval()
    words = start_text.split()
    state_h, state_c = model.init_state(batch_size=1)
    
    for _ in range(max_length):
        # Use the word2index dictionary with a fallback to <UNK> if the word is not found
        x = torch.tensor([[word2index.get(w, word2index['<UNK>']) for w in words]], dtype=torch.long)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index2word[word_index])
    
    return ' '.join(words)

# Streamlit app interface
st.title("Text Generation with Language Model")
st.write("Enter a text prompt, and the model will generate a continuation of the text.")

# Input box for user to type in a text prompt
user_input = st.text_input("Enter text prompt:", "Harry Potter is")

# Generate text based on user input
if user_input:
    with st.spinner('Generating text...'):
        generated_text = generate_text(model, user_input)
        st.success("Generated Text:")
        st.write(generated_text)
