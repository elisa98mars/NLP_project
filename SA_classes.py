import torch
from torch import nn

import warnings
warnings.filterwarnings("ignore")


class MyCollate:

    def __init__(self, pad_idx, maxlen):
        self.pad_idx = pad_idx
        self.maxlen = maxlen

    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        # Get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        # Pad them using pad_sequence method from pytorch.
        # source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx)

        padded_sequence = torch.zeros((self.maxlen, len(batch)), dtype = torch.int)

        for idx, text in enumerate(source):
            if len(text) > self.maxlen:
                padded_sequence[:, idx] = source[idx][: self.maxlen]
            else:
                padded_sequence[:len(source[idx]), idx] = padded_sequence[:len(source[idx]), idx] + source[idx]

        # Get all target indexed sentences of the batch
        target = [item[1] for item in batch]

        target = torch.tensor(target, dtype = torch.float32).reshape(-1)
        return padded_sequence, target
    

class config:
    warnings.filterwarnings("ignore", category = UserWarning)
    IMG_SIZE = (224,224)
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    FOLDS = 5
    SHUFFLE = True
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 15
    EMB_DIM = 100
    MAX_LEN = 20
    MODEL_PATH = "./Models/MyModel.pt"


class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, embedding_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer  # Initialize the embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)  # Bidirectional LSTM layer
        self.fc1 = nn.Linear(2 * hidden_dim, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, output_dim)  # Fully connected layer 2
        self.dropout = nn.Dropout(0.3)  # Dropout layer to prevent overfitting
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for binary classification

    def forward(self, text):
        max_len, N = text.shape  # Get the shape of the input text
        hidden = torch.zeros((2, N, self.hidden_dim), dtype=torch.float)  # Initialize hidden state
        memory = torch.zeros((2, N, self.hidden_dim), dtype=torch.float)  # Initialize memory state
        hidden = hidden.to(config.DEVICE)  # Move hidden state to the specified device (GPU or CPU)
        memory = memory.to(config.DEVICE)
        embedded = self.embedding(text)
        output, hidden = self.lstm(embedded, (hidden, memory))

        y_pred = output[-1,:,:]
        y_pred = self.fc1(y_pred)
        y_pred = self.fc2(y_pred)
        y_pred = self.sigmoid(y_pred)

        return y_pred