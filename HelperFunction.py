import numpy as np
import re
import torch
from torch import nn
import nltk
from tqdm import tqdm
tqdm.pandas()
from SA_classes import config
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text, lemmatizer, stop_words):

    text = text.lower()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def word_tokenizer(text):
    text = text.lower()
    text = text.split()
    return text

    
def get_max_similarity(text, aspects, model):
    try:
        text = " ".join(text)
        similarity_scores = [(aspect, model.wv.n_similarity(text, aspect)) for aspect in aspects]
        max_aspect, max_score = max(similarity_scores, key=lambda x: x[1])
        return max_aspect, max_score
    except:
        return None, None

def embed(tokens, nlp):
    """Return the centroid of the embeddings for the given tokens.

    Out-of-vocabulary tokens are cast aside. Stop words are also
    discarded. An array of 0s is returned if none of the tokens
    are valid.

    """
    lexemes = (nlp.vocab[token] for token in tokens)

    vectors = np.asarray([
        lexeme.vector
        for lexeme in lexemes
        if lexeme.has_vector
        and not lexeme.is_stop
        and len(lexeme.text) > 1])
    
    if len(vectors) > 0:
        centroid = vectors.mean(axis=0)
    else:
        width = nlp.meta['vectors']['width']
        centroid = np.zeros(width)

    return centroid

def predict(doc, nlp, neigh, label_names):

    tokens = doc.split(' ')
    centroid = embed(tokens, nlp)
    distances, indices = neigh.kneighbors([centroid])
    closest_label_index = indices[0][0]
    closest_distance = distances[0][0]

    return label_names[closest_label_index], closest_distance


#################################### Sentiment Analysis #################################################


def get_emb_layer_with_weights(target_vocab, emb_model, trainable=False):
    # Initialize an empty matrix and a counter
    weights_matrix = np.zeros((len(target_vocab), config.EMB_DIM))
    words_found = 0

    for i, word in enumerate(target_vocab):
        # Concatenate the word embedding for the current word from the embedding model
        weights_matrix[i] = np.concatenate([emb_model.wv[word]])
        words_found += 1

    print(f"Words found are: {words_found}")

    # Convert the weights matrix to a PyTorch tensor
    weights_matrix = torch.tensor(weights_matrix, dtype=torch.float32).reshape(len(target_vocab), config.EMB_DIM)
    # Create an embedding layer from the pre-trained weights matrix
    emb_layer = nn.Embedding.from_pretrained(weights_matrix)

    # Set the embedding layer's weights as trainable or non-trainable based on the 'trainable' flag
    if trainable:
        emb_layer.weight.requires_grad = True
    else:
        emb_layer.weight.requires_grad = False

    return emb_layer

def train_epochs(dataloader,model, loss_fn, optimizer):
    '''
    Function responsible for training the model for a single epoch.
    Computes the loss, backpropagates gradients, and updates the model's weights based on the specified optimizer.
    Keeps track of the training loss and the number of correct predictions during training.
    '''
    train_correct = 0
    train_loss = 0

    model.train()

    for review, label in tqdm(dataloader):

        review, label = review.to(config.DEVICE), label.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(review)
        output = output.reshape(-1)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*review.size(1)
        prediction = (output > 0.5).float()
        train_correct += (prediction == label).float().sum()

    return train_loss, train_correct

def val_epochs(dataloader, model, loss_fn):
    '''
    Used for validation during the training process, but it doesn't perform backpropagation or weight updates.
    Instead, it computes the validation loss and keeps track of the number of correct predictions.
    '''
    val_correct = 0
    val_loss = 0

    model.eval()

    for review, label in dataloader:

        review, label = review.to(config.DEVICE), label.to(config.DEVICE)

        output = model(review)
        output = output.reshape(-1)

        loss = loss_fn(output, label)

        val_loss += loss.item()*review.size(1)
        prediction = (output > 0.5).float()
        val_correct += (prediction == label).float().sum()

    return val_loss, val_correct


#################################### Inference #################################################


## These 2 functions help preprocess text data for input into a neural network

# Converts text into a list of numerical tokens using the vocabulary, adding special tokens like <SOS> and <EOS>
def numericalize(text, dataset):
    numerialized_source = []
    numerialized_source = [dataset.source_vocab.stoi["<SOS>"]]
    numerialized_source += dataset.source_vocab.numericalize(text)
    numerialized_source.append(dataset.source_vocab.stoi["<EOS>"])

    return numerialized_source

# Ensures that the input sequence is of a fixed length by either truncating or zero-padding the sequence as needed.
def padding(source):
    padded_sequence = torch.zeros(config.MAX_LEN, 1, dtype = torch.int)
    source = torch.tensor(source)

    if len(source) > config.MAX_LEN:
        padded_sequence[:, 0] = source[: config.MAX_LEN]
    else:
        padded_sequence[:len(source), 0] = padded_sequence[:len(source), 0] + source

    return padded_sequence

# Function used for processing text data before performing inference with a neural network
def infer_processing(text, lemmatizer, stop_words, dataset):
    text = clean_text(text, lemmatizer, stop_words)
    text = numericalize(text,dataset)
    text = padding(text)
    return text

# Define a function to determine the new value for each row
def determine_new_value(row):
    values_to_check = [row['fasttext_list'], row['fasttext_list2'], row['prediction'], row['aspect_spacy']]
    unique_values = set(values_to_check)  # Get unique values in the list

    if 'general' in unique_values and len(unique_values) == 1:
        return 'general'
    else:
        unique_values.discard('general')  # Remove 'general' if it's present
        return ', '.join(unique_values)