import pickle
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Load your CSV file into a DataFrame
df = pd.read_csv('datasets.csv')

output_answers = df['output']

# Define a function to load preprocessed data
def load_preprocessed_data():
    with open('preprocessed_inputs.pkl', 'rb') as file:
        return pickle.load(file)
        
# Now, you can use the loaded preprocessed data in your functions
preprocessed_inputs = load_preprocessed_data()

# Define a function to get the output based on user input
def get_output(user_input):
    user_input = nlp(user_input)

    # Find the most similar input in the dataset
    similarity_scores = [user_input.similarity(preprocessed_input) for preprocessed_input in preprocessed_inputs]
    most_similar_index = similarity_scores.index(max(similarity_scores))

    # Return the corresponding output
    return output_answers[most_similar_index]


# Example usage
while True:
    user_input = input("Type your instruction (type 'exit' to end): ")

    if user_input.lower() == 'exit':
        break
    output_text = get_output(user_input)
    print(output_text)