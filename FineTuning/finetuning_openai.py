"""
pip install openai
pip install openai[datalib]
pip install urllib3
pip install python-dotenv
pip install tiktoken
"""
import io
import os
import time
from openai import AzureOpenAI

from dotenv import load_dotenv
from pathlib import Path
import openai
import pandas as pd

# Load .env from the FineTurning folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Helper functions:
import json
import tiktoken  # for token counting
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base")


# input_file=formatted_custom_support.json ; output_file=output.jsonl
def json_to_jsonl(input_file, output_file):

    # Open JSON file
    f = open(input_file)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # produce JSONL from JSON
    with open(output_file, "w") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")


def check_file_format(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call") for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


# convert to jsonl from json
# we will use a different json dataset maybe from huggingface!
# use teacrafter.json file
script_dir = Path(__file__).parent
json_to_jsonl(script_dir / "teacrafter.json", script_dir / "output.jsonl")

# check file format:
data_path = script_dir / "output.jsonl"

# Load the dataset from:https://cookbook.openai.com/examples/chat_finetuning_data_prep
with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]


# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)


# Format validation
check_file_format(dataset)


# Cost estimations
# Get the length of the conversation
conversation_length = []

for msg in dataset:
    messages = msg["messages"]
    conversation_length.append(num_tokens_from_messages(messages))

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096
TARGET_EPOCHS = 5
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)

if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(
    min(MAX_TOKENS_PER_EXAMPLE, length) for length in conversation_length
)
print(
    f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
)
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(
    f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
)

num_tokens = n_epochs * n_billing_tokens_in_dataset

# gpt-3.5-turbo	$0.0080 / 1K tokens -- need updates, use gpt4o-mini
cost = (num_tokens / 1000) * 0.0080
print(cost)

# Upload file once all validations are successful!
"""
print("\nUploading training file to Azure OpenAI...")
training_file = client.files.create(
    file=open(data_path, "rb"), purpose="fine-tune"
)
print(f"File uploaded successfully! File ID: {training_file.id}")
fileId = training_file.id

"""

# Wait for file to be processed
"""
print("Waiting for file to be processed...")
max_wait_time = 60  # Maximum wait time in seconds
start_time = time.time()
while time.time() - start_time < max_wait_time:
    file_status = client.files.retrieve(fileId)
    print(f"File status: {file_status.status}")
    if file_status.status == "processed":
        print("File processing complete!")
        break
    elif file_status.status == "error":
        print(f"Error processing file: {file_status}")
        exit(1)
    time.sleep(5)  # Wait 5 seconds before checking again
else:
    print("Timeout waiting for file to process")
"""
#fileId = "file-025506e724f345d2aa2c6032267acd86"
# == Next steps: Create a fine-tuned model ===
# Start the fine-tuning job
# After you've started a fine-tuning job, it may take some time to complete. Your job may be queued
# behind other jobs and training a model can take minutes or hours depending on the
# model and dataset size.

# Azure OpenAI supported models for fine-tuning (based on your instance):
# gpt-35-turbo-0125, gpt-4o-2024-08-06, gpt-35-turbo, gpt-4o, gpt-4.1-2025-04-14
"""
response = client.fine_tuning.jobs.create(
    training_file=fileId,  # Use the uploaded file ID
    model="gpt-4.1-2025-04-14",  # Using gpt-4.1 as requested
    hyperparameters={
        "n_epochs": n_epochs  # Use calculated epochs from dataset analysis
    },
)
print(f"Fine-tuning job created: {response.id}")
print(f"Status: {response.status}")
"""

job_id = "ftjob-10e748779e5040e0869db0b9e4995c82"

# # Retrieve the state of a fine-tune
# # Status field can contain: running or succeeded or failed, etc.
"""print("\nRetrieving fine-tuning job status...")
state = client.fine_tuning.jobs.retrieve(job_id)
print("\n=== Fine-Tuning Job Status ===")
print(f"Job ID: {state.id}")
print(f"Status: {state.status}")
print(f"Model: {state.model}")
print(f"Created at: {state.created_at}")
print(f"Training file: {state.training_file}")
if hasattr(state, 'finished_at') and state.finished_at:
    print(f"Finished at: {state.finished_at}")
if hasattr(state, 'fine_tuned_model') and state.fine_tuned_model:
    print(f"Fine-tuned model: {state.fine_tuned_model}")
if hasattr(state, 'error') and state.error:
    print(f"Error: {state.error}")
if hasattr(state, 'hyperparameters') and state.hyperparameters:
    print(f"Hyperparameters: {state.hyperparameters}")
print(f"\nFull state object:\n{json.dumps(state.model_dump(), indent=2)}")

if state.status != "succeeded":
    exit('Stopping here for now - uncomment next sections to proceed with testing the fine-tuned model')
# # once training is finished, you can retrieve the file in "result_files=[]"
# result_file = "file-6tZRoEV4SJ8fwjuWuPQpszzwYS"
result_file = state.result_files[0] if state.result_files else None

""" 

## HARDCODED
result_file = "file-31be4b3fdb824fa188ee271ac1941b66"
print(f"Result files: {result_file}")

if not result_file:
    exit("No result file found, exiting")

file_data = client.files.content(result_file)



# # its binary, so read it and then make it a file like object
file_data_bytes = file_data.read()
file_like_object = io.BytesIO(file_data_bytes)

# # now read as csv to create df
df = pd.read_csv(file_like_object)
#print(df)

fine_tuned_model= "gpt-4.1-2025-04-14.ft-10e748779e5040e0869db0b9e4995c82"
fine_tuned_model = "gpt-4-04-14"

print("\n" + "="*60)
print("IMPORTANT: Azure OpenAI Deployment Required")
print("="*60)
print(f"Fine-tuned model: {fine_tuned_model}")
print("\nTo use this model, you must:")
print("1. Go to Azure Portal -> Your OpenAI Resource -> Model deployments")
print("2. Click 'Create new deployment'")
print("3. Select your fine-tuned model from the dropdown:")
print(f"   {fine_tuned_model}")
print("4. Give it a deployment name (e.g., 'teacrafter-bot')")
print("5. Use the DEPLOYMENT NAME (not model name) in your code")
print("\nExample: If you name deployment 'teacrafter-bot', use:")
print("  model='teacrafter-bot'")
print("="*60)


## Testing and evaluating -- first use the main model and prompt it:
## make sure to change the messages to match the data set we fine-tuned our model with
""" 

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": "This is a customer support chatbot designed to help with common inquiries.",
        },
        {
            "role": "user",
            "content": "How do I change my tea preferences for the next shipment?",
        }
    ],
)
print("\n === Base model response===\n")
print(
    response.choices[0].message.content
)  # it should give us a random response or "I don't know..."
print("\n ========")
""" 


"""

response = client.chat.completions.create(
    model=fine_tuned_model,  # Use deployment name, not model name
    messages=[
        {
            "role": "system",
            "content": "This is a customer support chatbot designed to help with common inquiries.",
        },
        {
            "role": "user",
            "content": "How do I change my tea preferences for the next shipment?",
        }
    ],
)
print(
    f" ==== >> Fined-tuned model response: \n {response.choices[0].message.content}"
)  # we should get a coherent answer since we are now using a fine-tuned model!!!

"""

context = [
    {
        "role": "system",
        "content": """This is a customer support chatbot designed to help with common 
                                           inquiries for TeaCrafters""",
    }
]


def collect_messages(
    role, message
):  # keeps track of the message exchange between user and assistant
    context.append({"role": role, "content": f"{message}"})


def get_completion():
    try:
        response = client.chat.completions.create(
            model=fine_tuned_model, messages=context
        )

        print("\n Assistant: ", response.choices[0].message.content, "\n")
        return response.choices[0].message.content
    except openai.APIError as e:
        print(e.http_status)
        print(e.error)
        return e.error


# Start the conversation between the user and the AI assistant/chatbot
while True:
    collect_messages(
        "assistant", get_completion()
    )  # stores the response from the AI assistant

    user_prompt = input("User: ")  # input box for entering prompt

    if user_prompt == "exit":  # end the conversation with the AI assistant
        print("\n Goodbye")
        break

    collect_messages("user", user_prompt)  # stores the user prompt

print("\n=== Fine-tuning setup complete! ===")
print("Dataset validated successfully.")
print("To start fine-tuning:")
print("1. Uncomment the file upload section (line ~183)")
print("2. Uncomment the fine-tuning job creation section (line ~189)")
print("3. After training completes, uncomment the testing sections to use your fine-tuned model")