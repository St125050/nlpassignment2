# Text Generation with Language Model

This project demonstrates training a language model using a dataset and deploying the model through a Streamlit web application. The model is trained to generate text based on an input prompt.

## Project Description

The project involves:
1. Preprocessing text data from a dataset.
2. Training a language model using PyTorch.
3. Saving the trained model.
4. Deploying the model using a Streamlit web application.

## Setup Instructions

### Prerequisites

- Python 3.7+
- Required Python packages: `requests`, `nltk`, `numpy`, `torch`, `streamlit`

### Installation

1. **Clone the repository** (if applicable):
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required packages**:
    ```sh
    pip install requests nltk numpy torch streamlit
    ```

3. **Download NLTK data**:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Training the Model

1. **Run the training script**:
    Save the following code in a file named `train_model.py`:

    ```python
    import requests
    import nltk
    import re
    from nltk.corpus import stopwords
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

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

    print(f"Total sequences: {len(input_sequences)}")

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    # Save the model's state dictionary and vocabulary
    torch.save(model.state_dict(), 'model.pth')
    np.save('vocab.npy', vocab)
    np.save('word2index.npy', word2index)
    np.save('index2word.npy', index2word)
    print("Model and vocabulary saved successfully.")
    ```

    Run the script to train the model and save the necessary files:
    ```sh
    python train_model.py
    ```

## Running the Streamlit App

1. **Create the Streamlit app**:
    Save the following code in a file named `app.py`:

    ```python
    import streamlit as st
    import torch
    import torch.nn as nn
    import numpy as np

    # Load the vocabulary and mappings
    vocab = np.load('vocab.npy', allow_pickle=True).tolist()
    word2index = np.load('word2index.npy', allow_pickle=True).item()
    index2word = np.load('index2word.npy', allow_pickle=True).item()

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
    ```

2. **Run the Streamlit app**:
    Open a terminal, navigate to the directory where `app.py` is saved, and run the following command:
    ```sh
    streamlit run app.py
    ```

This will launch a Streamlit web application where you can enter a text prompt, and the model will generate a continuation of the text.

## Deployment

To deploy your Streamlit app, you can use platforms such as Streamlit Cloud, Heroku, or any other cloud service that supports Python applications. For more details on deploying Streamlit apps, refer to the [Streamlit deployment documentation](https://docs.streamlit.io/streamlit-cloud).

## Acknowledgments

- [Project Gutenberg](https://www.gutenberg.org/) for providing the dataset.
- [Streamlit](https://streamlit.io/) for the web application framework.
- [PyTorch](https://pytorch.org/) for the machine learning library.
- [NLTK](https://www.nltk.org/) for the natural language processing toolkit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
