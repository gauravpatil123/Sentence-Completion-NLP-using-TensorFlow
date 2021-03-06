"""
Predict:
    1. Loads the pre initialized tokenizer
    2. Loads the pre trained model from the saved file
    3. Initializes a seed text for sentence completion
    4. Sets the number of words to be predicted after the seed text
    5. Defines the predict function using the loaded model
    6. predicts and prints the predicted text on command line
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import DataPreprocessing as DP
import BuildModel as BM
import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)
PROCESSED_DATA = DP.processed_data
tokenizer = PROCESSED_DATA.get_tokenizer()
max_seq_len = PROCESSED_DATA.get_max_seq_len()

Model = BM.BuildModel()
model = Model.load_model("TrainedModel.h5")

SEED_TEXT = "To quit"
NUM_NEXT_WORDS = 10

def predict(seed_text, num_next_words):
    for _ in range(num_next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_seq_len - 1, padding = 'pre')
        predicted = model.predict_classes(token_list, verbose = 0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

PREDICT = predict(SEED_TEXT, NUM_NEXT_WORDS)
logging.info(PREDICT)
