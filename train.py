import tensorflow as tf
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

ds_path = "dataset/products.csv"
model_save_path = "./model"

EPOCHS = 10
VOCAB_SIZE = 2000
BATCH_SIZE = 64

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

# This is fixed.
EMBEDDING_DIM = 64

df = pd.read_csv(ds_path)
df.category = df.category.map(lambda x: x.split("|")[0].replace("-", " "))
labels = df.category.unique().tolist()
label_count = len(labels)
print("Label count:", label_count)
titles = df.title.values.tolist()

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="[OOV]")
tokenizer.fit_on_texts(titles)
X = pad_sequences(tokenizer.texts_to_sequences(titles), maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of data tensor:", X.shape)
# serializing with panda
Y = pd.get_dummies(df.category).values
print("Shape of label tensor:", Y.shape)

X_train, X_test, Y_train, Y_test = sk.train_test_split(
    X, Y, test_size=0.2, random_state=1
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=X.shape[1]
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2)
        ),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(label_count, activation="softmax"),
    ]
)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)


history = model.fit(
    X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2
)

print("Saving model...")
model.save(model_save_path)
print("Saved")

loss, accuracy = model.evaluate(X_test, Y_test)
print("Test set:\n Loss: {:0.3f}\n Accuracy: {:0.3f}".format(loss, accuracy))

saved_model = tf.keras.models.load_model(model_save_path)


def predict(text, model):
    inp = [text]
    seq = tokenizer.texts_to_sequences(inp)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = labels[np.argmax(model.predict(padded))]
    return prediction


text = "خودکار استابیلو مدل 818M بسته 10عددی"

print(predict(text, saved_model))
