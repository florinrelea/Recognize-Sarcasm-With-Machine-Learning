import json
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = []
labels = []
urls = []

for line in open('resources/Sarcasm_Headlines_Dataset_v2.json', 'r'):
  item = json.loads(line)

  sentences.append(item['headline'])
  labels.append(item['is_sarcastic'])
  urls.append(item['article_link'])

vocab_size = 10_000
training_size = 20_000
padding_type='post'
max_length=100
trunc_type='post'
embedding_dim = 16

training_sentences = np.array(sentences[0:training_size])
testing_sentences = np.array(sentences[training_size:])
training_labels = np.array(labels[0:training_size])
testing_labels = np.array(labels[training_size:])

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<00V>')
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

def check_if_sentence_is_sarcastic(input_sentences):
  converted_seqs = tokenizer.texts_to_sequences(input_sentences)
  padded = pad_sequences(converted_seqs, maxlen=max_length,
                         padding=padding_type,
                         truncating=trunc_type)
  return model.predict(padded)

prediction1 = check_if_sentence_is_sarcastic([
  'mark zuckerberg likes to make it rain with lawmakers',
])

print(prediction1)