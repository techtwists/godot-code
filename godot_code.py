import pickle
import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Load your CSV file into a DataFrame
df = pd.read_csv('datasets.csv')

#input_texts = df['instruction']
output_answers = df['output']

# Preprocess input and output texts
#preprocessed_inputs = [nlp(text) for text in input_texts]

preprocessed_outputs = [nlp(text) for text in output_answers]

# Save preprocessed data using Pickle
#with open('preprocessed_inputs.pkl', 'wb') as file:
#    pickle.dump((preprocessed_outputs), file)


# Save preprocessed data using Pickle
with open('preprocessed_outputs.pkl', 'wb') as file:
    pickle.dump((preprocessed_outputs), file)

print("successfully saved")