# FinancialSentiment-ModelEvaluation

## Overview

This repository contains the code for a project that evaluates the performance of OpenAI and llama2 (4-bit) models in sentiment classification of financial_phrasebook headlines. In particular the project examines base/fine(instruct)-tuned GPT and llama2 models and compares their performance across one/few-shot prompting methods.

## Key Results

| **Model**                          | **Type**      | **Accuracy** | **F1** | **Accuracy Improvement (%)** | **F1 Improvement (%)** |
| ---------------------------------- | ------------- | ------------ | ------ | --------------------------- | ---------------------- |
| GPT 4                              | Zero-Shot     | 0.83         | 0.83   | —                           | —                      |
|                                    | Few-Shot      | 0.85         | 0.84   | 2.41                        | 1.2                    |
|                                    | Fine-Tuning   | —            | —      | —                           | —                      |
| GPT 3.5                            | Zero-Shot     | 0.78         | 0.77   | —                           | —                      |
|                                    | Few-Shot      | 0.83         | 0.83   | 6.41                        | 7.79                   |
|                                    | Fine-Tuning   | **0.88**     | **0.88** | **12.82**                  | **14.29**              |
| Llama-2 13B <br/> 4-bit            | Zero-Shot     | 0.75         | 0.74   | —                           | —                      |
|                                    | Few-Shot      | 0.75         | 0.74   | 0                           | 0                      |
|                                    | Fine-Tuning   | 0.8          | 0.79   | 6.67                        | 6.76                   |
| Llama-2 7B <br/> 4-bit             | Zero-Shot     | 0.47         | 0.46   | —                           | —                      |
|                                    | Few-Shot      | 0.43         | 0.41   | -8.51                       | -10.87                 |
|                                    | Fine-Tuning   | 0.65         | 0.6    | **38.3**                    | **30.43**              |


## Data Sources

- `all-data.csv`: This dataset contains financial headlines and was extracted from [Hugging Face's Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank).
- `df.csv`: This dataset is a cleaned version of `all-data.csv`, processed using `data_cleaning.py`.

## Directory Structure

- `data_preprocessing/`: Contains scripts for data cleaning and splitting the dataset into training and testing sets.
  - `data_cleaning.py`: Cleans the raw data.
  - `train_test_split.py`: Splits the dataset into training and test sets (the proportion can be specified in the config file).
- `openai/`: Contains scripts and configurations for OpenAI-based models.
  - `openai_finetuning.py`: Fine-tunes OpenAI models and outputs the tuned model id - plug it into the config file to run inference using the model.
  - `openai_classifier.py`: Classifies data using OpenAI models.
- `llama/`: Contains Jupyter Notebooks for working with Llama models.
  - `llama2_classifier.ipynb`: Notebook for Llama-based classification.
  - `llama2_finetuning.ipynb`: Notebook for fine-tuning Llama models.

## Configuration

The repository makes use of a configuration file `config.ini` to set various parameters. Here's a brief overview:

### [DEFAULT]
- `train_size`: 0.75
- `hf_token`: 
- `few_shot`: 0

### [openai_classifier]
- `model`: gpt-3.5-turbo
- `api_key`:
- `prompt`:

### [llama2_classifier]
- `model`: meta-llama/Llama-2-7b-chat-hf

## Usage

1. First, clone the repository.
2. Make sure to install all the required dependencies.
3. Update `config.ini` with your specific API keys, few/one-shot prompting and train size prefernces.
4. Run the data preprocessing scripts to clean and split the data.
5. Use OpenAI or Llama scripts for model fine-tuning and classification.
6. Examine your models performance in the 'results' section.

For more detailed information, refer to the individual Python scripts and Jupyter Notebooks within each directory.

## Contributions

Feel free to contribute to this repository by submitting pull requests or opening issues.

## Key Notes

### Prompts Used
Compared to OpenAI models, llama2 required more steering to produce reliable outputs. Hence prompts between the models differed slightly to include additional instructions:

#### OpenAI Models

- One-Shot "system" prompt:

\#\#\# Instructions: As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL].

- Few-Shot "system" prompt:

\#\#\# Instruction: As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL].

\# Example 1:
\#\#\# Headline: Consolidated pretax profit decreased by 69.2 % to EUR 41.0 mn from EUR 133.1 mn in 2007 .
\#\#\# Response: NEGATIVE

\# Example 2:
\#\#\# Headline: In 2007 , Huhtamaki will continue to invest in organic growth .
\#\#\# Response: NEUTRAL

\# Example 3:
\#\#\# Headline: MD Henning Bahr of Stockmann Gruppen praises the trend , since the chains become stronger and their decision-making processes more clear .
\#\#\# Response: POSITIVE

#### Llama2 Models

- One-shot prompt:

\#\#\# Instruction:
As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL].

\#\#\# Headline:
{headline}

\#\#\# Please respond with only one of the following labels: NEGATIVE, POSITIVE, or NEUTRAL.

\#\#\# Response: The sentiment expressed in the headline is

- Few-shot prompt:

\#\#\# Instruction: As a retail investor, you are presented with a financial headline. Your task is to classify the sentiment expressed in the headline using one of the following labels: [NEGATIVE, POSITIVE, NEUTRAL].

\# Example 1:
\#\#\# Headline: Consolidated pretax profit decreased by 69.2 % to EUR 41.0 mn from EUR 133.1 mn in 2007 .
\#\#\# Response: NEGATIVE


### Unexpected Model Output Handling

When models produce unexpected output, we adhere to the methodology outlined in Zhang, Yang & Liu (2023). Specifically, we substitute the label "NEUTRAL" in place of the unexpected output. For each model, the count of such substituted labels is stored in the `error_count` column within the `df_results` file.

