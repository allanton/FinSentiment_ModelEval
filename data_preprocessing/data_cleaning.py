import pandas as pd
import os

# Read CSV into a DataFrame
script_dir = os.path.dirname(__file__)
rel_path = "../data/all-data.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df = pd.read_csv(abs_file_path, encoding='Windows-1252')

df.columns = ['True_Label', 'Headline'] # Rename the columns
new_order = ['Headline', 'True_Label'] # Define the new order of the columns
df = df.reindex(columns=new_order) # Rearrange the columns according to the new order

df['True_Label'] = df['True_Label'].str.upper() # Capitalize the labels for consistency

# Function to remove both non-UTF-8 and non-standard ASCII characters from text
def remove_non_standard_and_utf8(text):
    return ''.join(char for char in str(text) if ord(char) < 128 and (ord(char) >= 32 and ord(char) != 127))

# Apply the function to clean all columns of the DataFrame
df = df.applymap(remove_non_standard_and_utf8)

# Drop the row that causes issues during fine-tuning
df = df[df['Headline'] != "As a result of the cancellation , the maximum increase of Citycon 's share capital on the basis of the convertible bonds decreased from EUR 23,383,927.80 to EUR 22,901,784.75 ."]
df.reset_index(drop=True, inplace=True)

abs_out_path = os.path.join(script_dir, "../data/df.csv")
df.to_csv(abs_out_path, index=False)
