# This script runs the OpenAI models on the test set and saves the results

import pandas as pd
import openai
import datetime
import pytz
import configparser
import time
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from openai_classes import OpenAI_Classifier

# Set the config path
script_dir = os.path.dirname(__file__)
abs_config_path = os.path.join(script_dir, '../config.ini')
config = configparser.ConfigParser()
config.read(abs_config_path)

# Read the config file
few_shot = config.getboolean('DEFAULT', 'few_shot') # Few shot learning
train_size = config.getfloat('DEFAULT', 'train_size') # Proportion of the train set
api_key = config.get('openai_classifier', 'api_key') # API key for the API
model = config.get('openai_classifier', 'model') # Model for the API
test_df_path = '../data/test_' + str(round(1-train_size,2)) + '.csv'
test_df_path = os.path.join(script_dir, test_df_path)

# Set the prompt
if few_shot == True:
    prompt = """### Instruction: As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL].

# Example 1:
### Headline: Consolidated pretax profit decreased by 69.2 % to EUR 41.0 mn from EUR 133.1 mn in 2007 .
### Response: NEGATIVE

# Example 2:
### Headline: In 2007 , Huhtamaki will continue to invest in organic growth .
### Response: NEUTRAL

# Example 3:
### Headline: MD Henning Bahr of Stockmann Gruppen praises the trend , since the chains become stronger and their decision-making processes more clear .
### Response: POSITIVE"""

else:
    prompt = config.get('openai_classifier', 'prompt') # Prompt for the API


# Read CSV into a DataFrame
df = pd.read_csv(test_df_path)
# df = df.iloc[:20, :]
openai.api_key = api_key

# Initiate the classifier class
openai_classifier = OpenAI_Classifier(
    model = model,
    prompt = prompt
    )

start_time = time.time()
# Apply the function to classify headlines and store the results in a new column
df['Predicted_Label'] = df['Headline'].apply(openai_classifier.classify_headline)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to complete.")

# Output the number of rows where NEUTRAL was subsituted due to errors/unrecognised output
# This is consistent with Zhang, Yang & Liu (2023)
print(f"Number of rows with substituted 'NEUTRAL' in the Predicted_Label column: {openai_classifier.error_count}")

# Define a dictionary to map the old values to the new values
mapping = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}

# Replace the values in the two columns using the mapping dictionary
df['True_Label'] = df['True_Label'].replace(mapping)
df['Predicted_Label'] = df['Predicted_Label'].replace(mapping)

# Calculate accuracy
accuracy = accuracy_score(df['True_Label'], df['Predicted_Label'])
print(f"Accuracy: {accuracy}")
# Calculate precision
precision = precision_score(df['True_Label'], df['Predicted_Label'], average='weighted')
print(f"Precision: {precision}")
# Calculate recall
recall = recall_score(df['True_Label'], df['Predicted_Label'], average='weighted')
print(f"Recall: {recall}")
# Calculate F1 score
f1 = f1_score(df['True_Label'], df['Predicted_Label'], average='weighted')
print(f"F1 score: {f1}")

#Store the results in the dataframe
df_results_path = os.path.join(script_dir, '../results/df_results.csv')
df_results = pd.read_csv(df_results_path)
model = openai_classifier.model
prompt = openai_classifier.prompt
error_count = openai_classifier.error_count
bst = pytz.timezone('Europe/London')
now = datetime.datetime.now(bst)
formatted_time = now.strftime('%d/%m/%Y/%H:%M')


df_results = pd.concat([
    df_results,
    pd.DataFrame({
        'Model': model,
        'Test_Size': round(1-train_size,2),
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Prompt': prompt,
        'Error Count': error_count,
        'DateTime': formatted_time,
        'Few_Shot': few_shot
    }, index=[0])
], ignore_index=True)

df_results.to_csv(df_results_path, index=False)


print('stoooop')


