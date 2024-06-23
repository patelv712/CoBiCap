# -*- coding: utf-8 -*-
"""
### Note
pre-processing and result generation code heavily draws from tutorial at https://www.kaggle.com/code/zohaib123/image-caption-generator-using-cnn-and-lstm
"""

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input , Dense , LSTM , Embedding , Dropout , add
from nltk.translate.bleu_score import corpus_bleu

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'flickr8k:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F623289%2F1111676%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240426%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240426T205942Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D78f08d617066783d96ad0e5b2b6aec71a7f75c6775ef902206e6a2294d9ee12cdd435ca7060d040e78b32583b45ab6cb0bd68de06bc75b60bf35566262e1d12954bf914f8e347542af5dca808cf9fafe8e34fe5ec4bca6aedaca58c275c158f1c4de0a9611daa4ba0e8d3e493b9404f62198df14a95ac71c5cfdcc51467c13078a9c6c646cae62a946f6e09c20eac91d7fce01d9a96b80aae765ea52dbacc7d342eba25bccd3568851726eb78d2fff82e24ed4a00748dbeb6f37c360b206fbfd0adeb88be6288096c9ce7300715f903e9afab8bb3f71e891ccb0c7bbdfd7b66a968dcd81ebbf6c5971a3a064c53e0bdd446f795065739d260e72181b3b1e7670'
KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'
BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/working'


# script to download data - from kaggle
for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory = data_source_mapping.split(':')[0]
    download_url_encoded = data_source_mapping.split(':')[1]
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
        total_length = fileres.headers['content-length']
        dl = 0
        data = fileres.read(CHUNK_SIZE)
        while len(data) > 0:
            dl += len(data)
            tfile.write(data)
            sys.stdout.flush()
            data = fileres.read(CHUNK_SIZE)
        if filename.endswith('.zip'):
            with ZipFile(tfile) as zfile:
            zfile.extractall(destination_path)
        else:
            with tarfile.open(tfile.name) as tarfile:
            tarfile.extractall(destination_path)


# we use VGG16 model to run feature extraction
# on the images from the above kaggle dataset
model = VGG16()
model = Model(inputs = model.inputs , outputs = model.layers[-2].output)
print(model.summary())


# feature extraction code blacboxed from above kaggle notebook
features = {}
directory = os.path.join(BASE_DIR, 'Images')
for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


# preprocessing code for captions
# draws from kaggle notebook above
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# cleaning up the captions
def clean(mapping):
    for _, captions in mapping.items():
        for i in range(len(captions)):
            captions[i] = "startseq ".join([word for word in captions[i].lower().replace('[^A-Za-z]', '').replace('\s+', ' ').split() if len(word)>1]) + " endseq"

# tokenize and get test train split
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


# This function is used to create data batches
# that the model will use to train on
# iterates over data_keys to process caption
# and create sequences
# ultimately outputs batches

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1 = list()
    X2 = list()
    y = list()
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1 = np.array(X1)
                X2 = np.array(X2)
                y = np.array(y)
                yield [X1, X2], y
                X1 = list()
                X2 = list()
                y = list()
                n = 0

"""# Vanilla-LSTM

Our Vanilla-LSTM model has a dual-input encoder-decoder architecture.
There are two distinct input pathways, one for image features and one for the textual data. 
Image input is vectorized in 4096x4096 format. 
This data was regularized with dropout=0.4 to avoid the risk of overfitting. 
The dense layer and the ReLU activation are critical in detecting non-linear patterns in the images. 
The caption text is processed by an embedding layer prior to getting mapped to 256-dimension. 
A LSTM layer then sequences through embedded words. 
The decoder brings these two pathways together, 
using a dense layer and ReLU activation which creates a probabilistic 
distribution for the next word in the generated caption.
Vanilla-LSTM employs categorical cross-entropy loss function and the Adam optimizer.

"""

i1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(i1)
fe2 = Dense(256, activation='relu')(fe1)
i2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)
d1 = add([fe2, se3])
d2 = Dense(256, activation='relu')(d1)
o = Dense(vocab_size, activation='softmax')(d2)
model = Model(inputs=[i1, i2], outputs=o)
model.compile(loss='categorical_crossentropy', optimizer='adam')

"""# Bi-LSTM

In the BI-LSTM model, we introduce the addition of three bidirectional LSTM layers instead of the singular LSTM layer.

Bi-LSTM builds off Vanilla-LSTM by adding Bidirectional LSTMs. 
Bidirectional LSTMs can process textual data both in the forward and backward direction, 
providing more context and allowing for more complex text understanding and generation. 
The LSTM layer from Vanilla-LSTM has been replaced with a series of three Bidirectional LSTM layers, 
each having output dimension 512 due to the two 256-dimensional compositional LSTMs that comprise the Bidirectional LSTM.

We hope to generate higher quality captions with the additional context gained from being able 
to process the sequential data in both the forward and backward direction.
"""

i1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(i1)
fe2 = Dense(256, activation='relu')(fe1)
i2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
se2 = Dropout(0.4)(se1)
se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)
se4 = Bidirectional(LSTM(256, return_sequences=True))(se3)
se5 = Bidirectional(LSTM(256))(se4)
d1 = Add()([fe2, se5])
d2 = Dense(256, activation='relu')(d1)
o = Dense(vocab_size, activation='softmax')(d2)
model = Model(inputs=[i1, i2], outputs=o)
model.compile(loss='categorical_crossentropy', optimizer='adam')


"""
In the process of creating a Bi-LSTM-based model, we also experimented with a variety of different architectures
to see what worked best.
"""



####################
# Experiment 2 - Dropout #
####################
i1 = Input(shape=(4096,))
"""
We tried increasing the dropout here to better regularize and see if that decreases
overfitting
"""
fe1 = Dropout(0.5)(i1)
fe2 = Dense(256, activation='relu')(fe1)
i2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
se2 = Dropout(0.5)(se1)
se3 = Bidirectional(LSTM(256))(se2)
d1 = Add()([fe2, se3])
d2 = Dense(256, activation='relu')(d1)
o = Dense(vocab_size, activation='softmax')(d2)
model2 = Model(inputs=[i1, i2], outputs=o)
model2.compile(loss='categorical_crossentropy', optimizer='adam')

"""
Analysis

The model did not perform as well, however as expected,
the new dropout rate did help with overfitting in our case.

"""


####################
# Experiment 3 - Loss Function #
####################

"""
In this experiment, we tried using a different loss function.
We used sparse_categorical_crossentropy instead of categorical_crossentropy.
This loss function typically does well when working with large vocabularies.
"""

i1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(i1)
fe2 = Dense(256, activation='relu')(fe1)
i2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
se2 = Dropout(0.4)(se1)
se3 = Bidirectional(LSTM(256))(se2)
se4 = Bidirectional(LSTM(256))(se3)
se5 = Bidirectional(LSTM(256))(se4)
d1 = Add()([fe2, se5])
d2 = Dense(256, activation='relu')(d1)
o = Dense(vocab_size, activation='softmax')(d2)
model3 = Model(inputs=[i1, i2], outputs=o)
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

"""
Analysis

With this new loss function, the model was computationally
stronger in that it was faster to deal with the large vocab.
However, there were no clear results that shows that accuracy
is better with this loss function.
"""


####################
# Experiment 4 - Residual Connections ??? #
####################

"""
In this experiment, we added residual connections to our LSTM layers.
This can help reduce the issue of vanishing gradient. Gradients
can much more easily transfer thorugh the network with residual connections.
Thus, training speed and convergence will likely be improved.
"""

i1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(i1)
fe2 = Dense(256, activation='relu')(fe1)
i2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
se2 = Dropout(0.4)(se1)
se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)
se2_i = Dense(512, activation='relu')(se2)
se4 = Add()([se2_i, se3])
se5 = Bidirectional(LSTM(256, return_sequences=True))(se4)
se5_i = Dense(256, activation='relu')(se5)
d1 = Add()([fe2, se5])
d2 = Dense(256, activation='relu')(d1)
o = Dense(vocab_size, activation='softmax')(d2)
model4 = Model(inputs=[i1, i2], outputs=o)
model4.compile(loss='categorical_crossentropy', optimizer='adam')

"""
Analysis

Training speed and convergence were indeed optimized
with this particular LSTM variation.
BLEU-1: 0.568576
BLEU-2: 0.338630
"""



# training code heavily draws from kaggle notebook cited above
# TRAINING
epochs = 20
# tested with epochs 10 for efficient training time
epochs = 10
# did not notice major difference in output
batch_size = 32
steps = len(train) // batch_size
for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)


# RESULTS  - draws heavily from kaggle notebook cited above


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# use model to predict the caption and get
# caption sequence based on start and end token
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None: break
        in_text += " " + word
        if word == 'endseq': break
    return in_text

actual, predicted = list(), list()


# calculate bleu metrics for evaluation
for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    actual.append(actual_captions)
    predicted.append(y_pred)

print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# Testing on image
# TODO: custom image captioning code goes here
generate_caption("101669240_b2d3e7f17b.jpg")