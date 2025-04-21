import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

texts = [
    "I love this product!", "Absolutely fantastic experience!",
    "It was okay, nothing special.", "Not too good, not too bad.",
    "Worst product I have ever used!", "I hated it completely!",
    "Great value for the money!", "Neutral feelings about this item.",
    "Terrible service!", "I’m quite satisfied."
]
labels = [2, 2, 1, 1, 0, 0, 2, 1, 0, 2]

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = 11
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

labels = to_categorical(labels, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=max_length))
model.add(SimpleRNN(32))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")


def predict_sentiment(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    results = []
    for i, text in enumerate(texts):
        predicted_class = np.argmax(predictions[i])
        results.append(
            f"Text: {text}\nSentiment: {sentiment_labels[predicted_class]}\n")
    return results


new_texts = ["I really enjoyed this product",
             "This was a terrible experience",
             "It was okay I guess", "I hate you"]
results = predict_sentiment(new_texts)
for result in results:
    print(result)
print(f"\nTest Accuracy: {accuracy:.4f}")
