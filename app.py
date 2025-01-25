import streamlit as st
import torch
import torch.nn as nn

# Define the model class (this should match your trained model's class)
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

# Load the trained model
model = LanguageModel(vocab_size=len(vocab), embedding_dim=50, hidden_dim=100)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the text generation function
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
st.write("Enter a text prompt and the model will generate a continuation of the text.")

# Input box for user to type in a text prompt
user_input = st.text_input("Enter text prompt:", "Harry Potter is")

# Generate text based on user input
if user_input:
    with st.spinner('Generating text...'):
        generated_text = generate_text(model, user_input)
        st.success("Generated Text:")
        st.write(generated_text)
