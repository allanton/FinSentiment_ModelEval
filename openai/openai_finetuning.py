# This script uploads the data to OpenAI for fine-tuning
# and produces the fine-tuned model

import os
import openai
import configparser
from time import sleep

# Set the config path
script_dir = os.path.dirname(__file__)
abs_config_path = os.path.join(script_dir, '../config.ini')
config = configparser.ConfigParser()
config.read(abs_config_path)
# Read the config
openai.api_key = config.get('openai_finetuning', 'api_key') # API key for the API
train_size = config.get('DEFAULT', 'train_size') # Number of rows to test the API
train_df_path = "../data/train_" + train_size + ".jsonl"
train_df_path = os.path.join(script_dir, train_df_path)

# Create a new fine-tuning event
res = openai.File.create(
    file=open(train_df_path, "r"),
    purpose='fine-tune'
)
file_id = res["id"]
print(f"!!! The created file id is {file_id}. Store this for later use. !!!")

# Create a new fine-tuning job
res = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)
job_id = res["id"]
print(f"!!! The created job id is {job_id}. Store this for later use. !!!")

# Wait for the fine-tuning job to complete
while True:
    res = openai.FineTuningJob.retrieve(job_id)
    if res["finished_at"] != None:
        break
    else:
        print(".", end="")
        sleep(100)

# Retrieve the fine-tuned model
ft_model = res["fine_tuned_model"]
print(f"!!! The created fine-tuned model id is {ft_model}. Store this for later use. !!!")