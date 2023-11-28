import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
import pickle

# Load your CSV file into a DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/techtwists/godot-code/main/datasets.csv')
df = df[:100]

# Modify the training data to include '<start>' and '<end>' tokens
df['output'] = '<start> ' + df['output'] + ' <end>'

input_texts = df['instruction'].values
output_texts = df['output'].values

# Tokenize the input sequences
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(input_texts)
input_sequences = tokenizer_input.texts_to_sequences(input_texts)
max_input_length = max(len(seq) for seq in input_sequences)

# Tokenize the output sequences
tokenizer_output = Tokenizer()
tokenizer_output.fit_on_texts(output_texts)
output_sequences = tokenizer_output.texts_to_sequences(output_texts)
max_output_length = max(len(seq) for seq in output_sequences)

# Pad sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
decoder_inputs = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# Define the model
embedding_dim = 256
hidden_units = 512

encoder_input = Input(shape=(max_input_length,))
encoder_embedding = Embedding(len(tokenizer_input.word_index) + 1, embedding_dim, input_length=max_input_length)(encoder_input)
encoder_lstm = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_input = Input(shape=(max_output_length,))
decoder_embedding = Embedding(len(tokenizer_output.word_index) + 1, embedding_dim, input_length=max_output_length)(decoder_input)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(tokenizer_output.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# Use mixed precision for better performance on compatible hardware
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Save the tokenizers
with open('tokenizer_input.pkl', 'wb') as f:
    pickle.dump(tokenizer_input, f)

with open('tokenizer_output.pkl', 'wb') as f:
    pickle.dump(tokenizer_output, f)

model.fit([encoder_inputs, decoder_inputs], decoder_inputs, epochs=10, batch_size=8, validation_split=0.2, callbacks=[checkpoint])
