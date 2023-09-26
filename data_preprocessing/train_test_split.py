# This script splits the data into train and test sets,
# formats the train split for openai finetuning,
# and saves the splits in the data folder.

import csv
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import configparser
import os


# Set the directory path and read in the config file
script_dir = os.path.dirname(__file__)
abs_config_path = os.path.join(script_dir, '../config.ini')
config = configparser.ConfigParser()
config.read(abs_config_path)

# Read the config file
train_size = config.getfloat('DEFAULT', 'train_size') # Size of the training split

# Read in the CSV file
abs_data_path = os.path.join(script_dir, '../data/df.csv')
df = pd.read_csv(abs_data_path)

# Split the DataFrame into train and test sets
train, test = train_test_split(df, train_size=train_size, random_state=42)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Transform the train set to JSONL format for GPT fine-tuning
output_json_list = []
for index, row in train.iterrows():
    label = row['True_Label']
    text = row['Headline']
    json_obj = {
        "messages": [
            {
                "role": "system",
                "content": "### Instructions: As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL]."
            },
            {
                "role": "user",
                "content": f"### Headline: {text}"
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }
    output_json_list.append(json_obj)

# Save the output as a JSONL file for GPT fine-tuning
train_json_file_path = '../data/train_' + str(train_size) + '.jsonl'
abs_train_json_file_path = os.path.join(script_dir, train_json_file_path)
with open(abs_train_json_file_path, 'w', encoding='utf-8') as jsonfile:
    for obj in output_json_list:
        jsonfile.write(json.dumps(obj) + '\n')

# Save the test and train in CSV format for llama
train_output_path = '../data/train_' + str(train_size) + '.csv'
abs_train_output_path = os.path.join(script_dir, train_output_path)
test_output_path = '../data/test_' + str(round(1 - train_size,2)) + '.csv'
abs_test_output_path = os.path.join(script_dir, test_output_path)
train.to_csv(abs_train_output_path, index=False)
test.to_csv(abs_test_output_path, index=False)