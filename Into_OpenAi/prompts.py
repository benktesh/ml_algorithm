from openai import AzureOpenAI  # must install openai package
import os
from pathlib import Path

from dotenv import load_dotenv

# Load Azure OpenAI settings from the workspace root .env

os.system("cls" if os.name == "nt" else "clear")



env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Azure OpenAI uses deployment name for the model parameter
model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")


# == few-shot learning
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a translator."},
        {
            "role": "user",
            "content": """ Translate these sentences: 
            'Hello' -> 'Namaste', 
            'Goodbye' -> 'Feri bhaetaula'. 
            '.
             Now translate: 'Thank you'.""",
        },
    ],
)
#print(completion.choices[0].message.content)


# Direct prompt example with openai / Zero-shot prompting
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the ISO3 country code of France?"},
    ],
)

#print(completion.choices[0].message.content)

#print("Making chain of thoughts")
# == Chain of thought ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a math tutor."},
        {
            "role": "user",
            "content": "Solve this math problem step by step: If John has 5 apples and gives 2 to Mary, how many does he have left?",
        },
    ],
)
#print(completion.choices[0].message.content)

# == Instructional prompts
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You a Digitalization of Carbon market expert.", 
        },
        {
            "role": "user",
            "content": "Write a 100-word summary of problem in carbon markets that digitalization solves",
        },
    ],
)
#print(completion.choices[0].message.content)

# == Role-playing prompts ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a character in a fantasy novel."},
        {
            "role": "user",
            "content": "Describe the setting of the story.",
        },
    ],
)
#print(completion.choices[0].message.content)

# == Open-ended prompt ==
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a philosopher."},
        {
            "role": "user",
            "content": "What is the meaning of life?",
        },
    ],
)
#print(completion.choices[0].message.content)



# == temperature and top-p sampling
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a creative tagline for a coffee shop."},
    ],
    #temperature=0.9,  # controls the randomness of the output
    top_p=0.1,  # controls the diversity of the output
)
#print(completion.choices[0].message.content)

# === Combining techniques ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a travel blogger."},
        {
            "role": "user",
            "content": "Write a 500-word blog post about your recent trip to Paris. Make sure to give a step-by-step itinerary of your trip.",
        },
    ],
    stream=True,
    # top_p=0.9,
)
#print(completion.choices[0].message.content)

for chunk in completion:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content or "", end="")