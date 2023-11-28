import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
# Load the trained model
model = load_model('model.h5')

# Load the tokenizers
with open('tokenizer_input.pkl', 'rb') as f:
    tokenizer_input = pickle.load(f)

with open('tokenizer_output.pkl', 'rb') as f:
    tokenizer_output = pickle.load(f)

# Function to generate predictions
def generate_output_sequence(input_sequence, model, tokenizer_input, tokenizer_output, max_input_length, max_output_length):
    # Tokenize the input sequence
    input_sequence = tokenizer_input.texts_to_sequences([input_sequence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_input_length, padding='post')

    # Initialize the decoder input with a start token
    target_seq = np.zeros((1, max_output_length))
    start_token = tokenizer_output.word_index.get('<start>', None)

    if start_token is not None:
        target_seq[0, 0] = start_token
    else:
        raise ValueError('"<start>" token not found in tokenizer_output word_index.')

    # Generate the output sequence
    for i in range(1, max_output_length):
        output_probs = model.predict([input_sequence, target_seq])[0, i - 1, :]
        predicted_token_index = np.argmax(output_probs)
        target_seq[0, i] = predicted_token_index

        end_token_index = tokenizer_output.word_index.get('<end>', None)
        if end_token_index is not None and predicted_token_index == end_token_index:
            break

    # Convert the predicted sequence back to text
    predicted_sequence = [tokenizer_output.index_word[index] for index in target_seq[0] if index > 0]
    return ' '.join(predicted_sequence)

# Example usage
max_input_length = 80
max_output_length = 187
input_instruction = "your input instruction here"
predicted_output = generate_output_sequence(input_instruction, model, tokenizer_input, tokenizer_output, max_input_length, max_output_length)
print("Predicted Output:", predicted_output)
