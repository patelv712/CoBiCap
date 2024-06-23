"""
### Note
Data setup and general methodology was inspired by https://www.kaggle.com/code/saeedghamshadzai/image-captioning-transformers-flickr8k/notebook
and https://www.kaggle.com/code/abubow/image-caption-generation-code
"""

import os
import warnings
import re
from keras import layers
from keras.applications import efficientnet
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm_notebook  # progress bars
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go  # Will use this later for interactive plots
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import TextVectorization
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import numpy as np

# Set up some global variables
IMG_DIR = "/kaggle/input/flickr8k/Images"
CAPTION_FILE = "/kaggle/input/flickr8k/captions.txt"
IMG_DIMENSIONS = (299, 299)  # for EfficientNet
MAX_SEQ_LEN = 15  # Max length of captions
VOCAB_SIZE = 5000  # Number of words in vocabulary
EMBEDDING_DIM = 256  # Embedding dimension for words
FF_UNITS = 156  # Number of units in feed-forward layer
BATCH_SIZE = 256  # Batch size for training
NUM_EPOCHS = 10  # Number of epochs to train for

"""
First step was to take the Flickr8k dataset and preprocess it with steps such as caption normalization and length filtering.
The images also had to be preprocessed with steps such as resizing.
"""

def load_and_prepare_image(path, target_size):
    # Load the image from the path and resize it
    return load_img(path, target_size=target_size)

def clean_captions(captions):
    # Strip the start and end tokens from captions and trim whitespace
    # print(len(captions))
    return [caption.replace("<start>", "").replace("<end>", "").strip() for caption in captions]

def select_random_images(image_paths, num_samples):
    # Randomly select and return image paths
    return np.random.choice(image_paths, size=num_samples, replace=False)

def visualize_results(dataset, predicted_captions, bleu_score_func, num_samples):
    selected_paths = select_random_images(list(dataset.keys()), num_samples)
    fig = plt.figure(figsize=(6, 20))  # Tall figure for displaying images and captions
    # fig = plt.figure(figsize=(3, 5))  # Smaller figure size for report screenshots
    image_index = 1

    for image_path in selected_paths:
        actual_captions = clean_captions(dataset[image_path])
        predicted_caption = predicted_captions[image_path]
        bleu_scores = bleu_score_func(actual_captions, [predicted_caption] * len(actual_captions))
        image = load_and_prepare_image(image_path, (199, 199, 3))  # Resize image for display

        # Display image
        ax_image = fig.add_subplot(num_samples, 2, image_index, xticks=[], yticks=[])
        ax_image.imshow(image)

        # image_index ++
        image_index += 1

        # show captions and BLEU scores
        ax_captions = fig.add_subplot(num_samples, 2, image_index)
        plt.axis('off')
        ax_captions.set_xlim(0, 1)
        ax_captions.set_ylim(0, len(bleu_scores))

        for i, score in enumerate(bleu_scores):
            ax_captions.text(0, i, f'The score is: {score:.2f}', fontsize=10)

        image_index += 1

    plt.tight_layout()
    plt.show()


def read_caption_file(file_path):
    """Reads the caption file and skips the header line."""
    with open(file_path) as f:
        return f.readlines()[1:]  # Skip the header line

def process_caption_line(line, skipped_images, caption_map):
    """Processes a single line of the caption file."""
    img_file, caption_text = line.strip().split(",", 1)
    img_file = os.path.join(IMG_DIR, img_file.strip())
    # print(img_file)

    caption_words = caption_text.strip().split()
    if len(caption_words) < 5 or len(caption_words) > MAX_SEQ_LEN:
        skipped_images.add(img_file)
        return None

    if (img_file.endswith("jpg") or img_file.endswith("png")) and img_file not in skipped_images:
        formatted_caption = "<start> " + caption_text.strip() + " <end>"
        # formatted_caption = "<start> " + caption_text + " <end>" # needed to add a strip to remove leading and trailing spaces
        return img_file, formatted_caption
    return None

def update_caption_map(img_file, formatted_caption, caption_map):
    """Updates the caption map with the processed caption."""
    if img_file in caption_map:
        caption_map[img_file].append(formatted_caption)
    else:
        caption_map[img_file] = [formatted_caption]

def load_caption_data(file_path):
    caption_lines = read_caption_file(file_path + "")
    caption_map = {}
    captions = []
    skipped_images = set()

    for line in caption_lines:
        result = process_caption_line(line, skipped_images, caption_map)
        if not result:
            # print("line was not in caption")
            continue
        else:
            img_file, formatted_caption = result
            update_caption_map(img_file, formatted_caption, caption_map)
            captions.append(formatted_caption)

    # Remove entries for images that should be skipped
    for img_file in skipped_images:
        caption_map.pop(img_file, None)

    return caption_map, captions

# Second visualization will be done after training model
visualize_results(test_set, predicted_captions, calculate_bleu_scores, num_samples=7)

def standardize_text(text):
    """Converts text to lowercase and removes punctuation and digits."""
    lowercase_text = tf.strings.lower(text)
    punctuation_and_digits = "!\"#$%&'()*+,-./:;=?@[\\]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase_text, f"[{re.escape(punctuation_and_digits)}]", "")

def configure_text_vectorizer(standardize_func):
    """Configures a TextVectorization layer."""
    return TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQ_LEN,
        standardize=standardize_func
    )

def create_image_augmenter():
    """Creates a sequential model for image augmentation."""
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3)
    ])

text_vectorizer = configure_text_vectorizer(standardize_text)
text_vectorizer.adapt(captions_list)

# print(len(caption_list)) # mistmatch error, take a look later;

image_augmenter = create_image_augmenter()

# Converting tensors to strings
processed_captions = [str(standardize_text(caption).numpy())[2:-1] for caption in captions_list]

# Assuming necessary global variables and preprocess_text are defined
def preprocess_text(text):
    lowercase_text = tf.strings.lower(text)
    # remove_chars = "!\"#$%&'()*+,-./:;=?@[\\]^_`{|}~" # needed to add numbers
    remove_chars = "!\"#$%&'()*+,-./:;=?@[\\]^_`{|}~1234567890" # used same list as before
    return tf.strings.regex_replace(lowercase_text, "[%s]" % re.escape(remove_chars), "")

def preprocess_captions(captions):
    """Process a list of captions through the preprocess_text function and convert to strings."""
    return [str(preprocess_text(caption).numpy())[2:-1] for caption in captions]

# new visaulizer needed for report data and comparison
# took previous visualizer and edited
def visualize_data(dataset, num_samples):
    fig = plt.figure(figsize=(10, 10))
    img_count = 1

    # Ensure we do not go out of index range
    sample_keys = list(dataset.keys())[100:100 + num_samples] if len(dataset) > 100 else list(dataset.keys())[:num_samples]

    for img_file in sample_keys:
        # Process captions for the image
        captions = preprocess_captions(dataset[img_file]) # call preprocessor
        # Load and prepare image
        img = load_img(img_file, target_size=(199, 199, 3))

        # Display the image
        ax_img = fig.add_subplot(num_samples, 2, img_count, xticks=[], yticks=[])
        ax_img.imshow(img)
        img_count += 1

        # Display the captions
        ax_captions = fig.add_subplot(num_samples, 2, img_count)
        plt.axis('off')
        ax_captions.plot()
        ax_captions.set_xlim(0, 1)
        # ax_captions.set_xlim(0, len(captions))
        ax_captions.set_ylim(0, len(captions))

        for i, caption in enumerate(captions):
            ax_captions.text(0, i, caption, fontsize=20)

        img_count += 1

    plt.tight_layout()
    plt.show()

# Assuming train_set is defined and properly populated
visualize_data(train_set, num_samples=7)

def get_caption_lengths(captions):
    """Calculate # of words in captions"""
    return [len(caption.split(' ')) for caption in captions]

def plot_caption_lengths(captions):
    """Plot distribution of caption lengths."""
    plt.figure(figsize=(15, 7), dpi=300)
    sns.set_style('darkgrid')

    caption_lengths = get_caption_lengths(captions)
    sns.histplot(x=caption_lengths, kde=True, binwidth=1)

    plt.title('Caption Length Distribution', fontsize=15)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

# Example usage assuming 'processed_captions' is defined
plot_caption_lengths(processed_captions)

def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # not sure if this is right
    img = tf.image.resize(img, IMG_DIMENSIONS)

    # img = tf.image.decode_jpeg(img) # channels?, error thrown

    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def vectorize_captions(captions):
    """Apply text vectorization to list of captions."""
    # print(f"Vectorizing {len(captions)} captions...")
    return text_vectorizer(captions)

def prepare_data(image_path, captions):
    """Prepare tuple of processed image and vectorized captions."""
    processed_image = preprocess_image(image_path)
    # print(f"Processed image shape: {processed_image.shape}")
    vectorized_captions = vectorize_captions(captions)
    # print(f"Vectorized captions shape: {vectorized_captions.shape}")
    return processed_image, vectorized_captions

def create_dataset(image_paths, caption_lists):
    """Create a tf.data.Dataset from image paths and captions."""
    # print(f"Creating dataset with {len(image_paths)} images and {len(caption_lists)} caption lists...")
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, caption_lists))
    dataset = dataset.shuffle(8 * BATCH_SIZE)  # Shuffle the dataset
    # print(f"Shuffled dataset size: {len(dataset)}")
    dataset = dataset.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Batching needed
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # print(f"Final dataset size (batched): {len(dataset)}")
    return dataset

# Assuming train_set and val_set are dictionaries with image paths as keys and lists of captions as values
train_dataset = create_dataset(list(train_set.keys()), list(train_set.values()))
val_dataset = create_dataset(list(val_set.keys()), list(val_set.values()))

# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Validation dataset size: {len(val_dataset)}")

"""
We decided that it makes the most sense to start with a pretrained CNN as we did not have the resources or the time to train a computer vision CNN from the ground up.
We were debating between using InceptionV3 and EfficientNet. Although InceptionV3 seems to perform better at a very large scale, EfficientNet, as the name suggest,
seems to outperform other CNNs at its complexity level, indicating that it is more resource-efficient and seems better for the image captioning task that doesn't
require extremely performant or complicated training or inference. That is why we decided to go with EfficientNetB0 CNN.
"""

def create_cnn_model(img_dimensions):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*img_dimensions, 3),
        include_top=False,
        weights="imagenet"
    )

    # print(f"Base model loaded: {base_model}")

    base_model.trainable = False
    x = layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    cnn_model = models.Model(inputs=base_model.input, outputs=x)
    # print(f"CNN model created: {cnn_model.summary()}")
    return cnn_model

"""
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TRANSFORMER CODE

We also applied positional encoding to the input of both the encoder and the decoder to ensure that we retained positional data about the image features and the words in the caption.
"""

class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.dense = layers.Dense(embedding_dim, activation="relu")
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        # print(f"Encoder input shape: {inputs.shape}")
        x = self.norm1(inputs)
        x = self.dense(x)
        attn_output = self.attention(x, x, x, training=training)
        x = self.norm2(x + attn_output)
        # print(f"Encoder output shape: {x.shape}")
        return x

class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_length, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.position_embeddings = layers.Embedding(input_dim=seq_length, output_dim=embedding_dim)
        self.embedding_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32))

    def call(self, inputs):
        # print(f"Positional embedding input shape: {inputs.shape}")
        positions = tf.range(tf.shape(inputs)[-1])
        tokens = self.token_embeddings(inputs) * self.embedding_scale
        positions = self.position_embeddings(positions)
        # print(f"Positional embedding output shape: {tokens.shape}")
        return tokens + positions

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

class TransformerDecoder(layers.Layer):
    def __init__(self, embedding_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embedding_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.5)

    def call(self, inputs, encoder_outputs, training=False, mask=None):
        # print(f"Decoder input shape: {inputs.shape}")
        attn1 = self.attention1(inputs, inputs, inputs, attention_mask=self.get_causal_attention_mask(inputs))
        out1 = self.norm1(inputs + attn1)
        attn2 = self.attention2(out1, encoder_outputs, encoder_outputs, attention_mask=mask)
        out2 = self.norm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout1(ffn_output, training=training)
        ffn_output = self.norm3(ffn_output + out2)
        # print(f"Decoder output shape: {ffn_output.shape}")
        return self.dropout2(ffn_output, training=training)

    def get_causal_attention_mask(self, inputs):
        seq_length = tf.shape(inputs)[1]
        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        return mask[tf.newaxis, :, :]

class ImageCaptioningModel(models.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, image_augmenter=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.image_augmenter = image_augmenter
        self.loss_tracker = models.metrics.Mean(name="loss")
        self.acc_tracker = models.metrics.Mean(name="accuracy")

    def call(self, inputs, training=False):
        images, captions = inputs
        if self.image_augmenter and training:
            images = self.image_augmenter(images)
        # print(f"Input images shape: {images.shape}")
        # print(f"Input captions shape: {captions.shape}")
        image_features = self.cnn_model(images)
        # print(f"Image features shape: {image_features.shape}")
        encoder_outputs = self.encoder(image_features, training=training)
        # print(f"Encoder outputs shape: {encoder_outputs.shape}")
        decoder_outputs = self.decoder(captions, encoder_outputs, training=training)
        # print(f"Decoder outputs shape: {decoder_outputs.shape}")
        return decoder_outputs

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

"""
TRANSFORMER CODE
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

cnn_model = create_cnn_model()
encoder = TransformerEncoder(embedding_dim=EMBEDDING_DIM, dense_dim=FF_UNITS, num_heads=2)
decoder = TransformerDecoder(embedding_dim=EMBEDDING_DIM, ff_dim=FF_UNITS, num_heads=3)
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_augmenter=image_augmenter)

# print(f"Caption model created: {caption_model.summary()}")

# Define the loss function with non-reduction at logit computation
cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Setup EarlyStopping to halt training when validation loss fails to improve
early_stopping_callback = keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

# Define a learning rate scheduler that linearly increases the learning rate during warmup period
class LinearWarmupLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate, warmup_steps):
        super().__init__()
        self.base_learning_rate = base_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_progress = global_step / self.warmup_steps
        warmup_learning_rate = self.base_learning_rate * warmup_progress
        is_warmup = global_step < self.warmup_steps
        return tf.where(is_warmup, warmup_learning_rate, self.base_learning_rate)

# Compute the number of training steps and warmup steps
num_training_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_training_steps // 15

# Initialize the learning rate schedule
learning_rate_schedule = LinearWarmupLearningRateSchedule(
    base_learning_rate=1e-4,
    warmup_steps=num_warmup_steps
)

# print(f"Learning rate schedule created with {num_warmup_steps} warmup steps")

# Compile the model with the optimizer using the defined learning rate schedule
caption_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate_schedule),
    loss=cross_entropy_loss
)

# print(f"Model compiled with Adam optimizer and learning rate schedule")

# Train the model with specified dataset, validation data, and callbacks
training_history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping_callback]
)

# print(f"Model training completed")

# Assuming vocabulary and text_vectorizer are already defined
vocabulary = text_vectorizer.get_vocabulary()
INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocabulary)}
MAX_DECODED_LENGTH = MAX_SEQ_LEN - 1
test_image_paths = list(test_set.keys())

# def preprocess_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, IMG_DIMENSIONS)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     return img

def generate_caption(image_path):
    image = preprocess_image(image_path)
    image = tf.expand_dims(image, 0)

    image_features = caption_model.cnn_model(image)

    # print(image_features[0:10])

    encoder_output = caption_model.encoder(image_features, training=False)

    decoded_caption = ["<start>"]

    for i in range(MAX_DECODED_LENGTH):
        tokenized_caption = text_vectorizer([decoded_caption])[:, :-1]
        mask = tf.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoder_output, training=False, mask=mask)
        sampled_token_index = tf.argmax(predictions[0, i, :], axis=-1).numpy()
        sampled_token = INDEX_TO_WORD[sampled_token_index]

        if sampled_token == "<end>":
            break

        decoded_caption.append(sampled_token)

    final_caption = ' '.join(decoded_caption).replace("<start>", "").replace("<end>", "").strip()

    return final_caption

# Initialize the dictionary to store the predicted captions
predicted_captions = {}

# Create a progress bar for the caption generation process
progress_bar = tqdm(total=len(test_image_paths), position=0, leave=True, colour='green')

for image_path in test_image_paths:
    try:
        # Generate caption for the current image
        caption = generate_caption(image_path)
        predicted_captions[image_path] = caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        predicted_captions[image_path] = "Error generating caption"
    finally:
        # Update the progress bar after each iteration
        progress_bar.update(1)

# Ensure the progress bar is closed after completion
progress_bar.close()

def calculate_bleu_scores(actual_captions, predicted_caption):
    # Assuming INDEX_TO_WORD is correctly defined, as per previous discussion
    processed_captions = []
    for caption in actual_captions:
        # Convert each token to a word using INDEX_TO_WORD if it's not a padding token (index 0)
        words = [INDEX_TO_WORD[idx] for idx in text_vectorizer(caption).numpy() if idx != 0]
        processed_captions.append(words)

    # Ensure predicted_caption is a list of tokens, not a single string
    predicted_caption_tokens = [predicted_caption.split()]  # Nested list for corpus_bleu

    # print(predicted_caption_tokens[0:5])

    # Calculate BLEU scores
    bleu1 = corpus_bleu([processed_captions], predicted_caption_tokens, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu([processed_captions], predicted_caption_tokens, weights=(0.5, 0.5, 0, 0))

    results = {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'Generated Caption': predicted_caption
    }

    return results

def visualize_results(dataset, predicted_captions, bleu_score_func, num_samples):
    # Ensure that you are selecting random paths in a reproducible manner if needed
    image_paths = list(dataset.keys())
    selected_paths = np.random.choice(image_paths, size=num_samples, replace=False)

    fig = plt.figure(figsize=(10, num_samples * 4))  # Adjust the size based on your preference

    for i, image_path in enumerate(selected_paths, 1):
        actual_captions = dataset[image_path]

        # print(len(actual_captions))

        actual_captions = [caption.replace("<start>", "").replace("<end>", "").strip() for caption in actual_captions]

        predicted_caption = predicted_captions.get(image_path, "No caption generated")
        bleu_scores = bleu_score_func(actual_captions, predicted_caption)  # Assume that the function is correctly defined

        # print(len(bleu_scores))

        image = load_img(image_path, target_size=(299, 299))  # Adjust target size as needed
        ax_image = fig.add_subplot(num_samples, 2, 2 * i - 1)
        ax_image.imshow(image)
        ax_image.axis('off')  # Hide axes
        ax_image.set_title("Image")

        ax_text = fig.add_subplot(num_samples, 2, 2 * i)
        plt.axis('off')
        text_display = f"Predicted: {predicted_caption}\n" + "\n".join([f"{k}: {v:.4f}" for k, v in bleu_scores.items()])
        ax_text.text(0.01, 0.5, text_display, verticalalignment='center', fontsize=12, wrap=True)

    plt.tight_layout()
    plt.show()

# Example usage
visualize_results(test_set, predicted_captions, calculate_bleu_scores, num_samples=7)