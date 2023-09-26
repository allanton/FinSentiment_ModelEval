# This script contains the OpenAI_Classifier class, which is used to classify headlines
# with OpenAI models.

import openai
from openai.error import APIConnectionError
from openai.error import RateLimitError
from openai.error import ServiceUnavailableError
import time


class OpenAI_Classifier:
    
    def __init__(self, model: str, prompt: str):
        self.model = model
        self.prompt = prompt
        self.error_count = 0
    
    # function to classify headlines with chatcompletion
    def classify_headline(self, headline: str):
        prompt = self.prompt
        # self.error_count = 0
        rate_limit_error_count = 0
        connection_error_count = 0               
        while True:  # Keep trying until successful or until reaching a retry limit
            try:  
                response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"### Headline: {headline}"}
                    ]
                )
                label = response.choices[0].message.content.strip()
                break  # Exit the loop if successful
            except APIConnectionError as e:
                print(f"Error communicating with OpenAI: {e}")
                label = "NEUTRAL"
                self.error_count += 1
                # connection_error_count += 1
                print(f"API Connection error count: {self.error_count}")
            except RateLimitError as e:
                print(f"Rate Limit Error: {e}")
                print("Waiting for 1 minute before retrying...")
                rate_limit_error_count += 1
                print(f"Rate Limit Error count : {rate_limit_error_count}")
                time.sleep(60)
            except ServiceUnavailableError as e:
                print(f"Service Unavailable Error: {e}")
                print("Waiting for 1 minute before retrying...")
                connection_error_count += 1
                print(f"Service unuavailable error count: {connection_error_count}")
                time.sleep(60)
            
            # Optionally, you can set a retry limit to prevent infinite loops
            if connection_error_count >= 10:  # For example, retry limit = 3
                print("Reached maximum number of retries. Exiting...")
                label = "NEUTRAL"
                self.error_count += 1
                break

        if label.upper() not in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            label = 'NEUTRAL'
            self.error_count += 1
            
        return label.upper()
    

        