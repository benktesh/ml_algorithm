"""
Check available models and deployments in Azure OpenAI
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the FineTuning folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

print("=== Checking Azure OpenAI Models ===\n")

try:
    # List available models
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
        if hasattr(model, 'capabilities'):
            print(f"    Capabilities: {model.capabilities}")
    print()
except Exception as e:
    print(f"Error listing models: {e}\n")

# Check fine-tuning jobs (to verify fine-tuning access)
try:
    jobs = client.fine_tuning.jobs.list(limit=1)
    print("✓ Fine-tuning API is accessible")
    print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
except Exception as e:
    print(f"✗ Fine-tuning API error: {e}")

print("\n=== Azure OpenAI Fine-tuning Requirements ===")
print("1. Your Azure OpenAI resource must be in a supported region:")
print("   - North Central US")
print("   - Sweden Central") 
print("   - Switzerland West")
print("\n2. Supported base models for fine-tuning:")
print("   - gpt-35-turbo (0613)")
print("   - gpt-35-turbo (1106)")
print("   - babbage-002")
print("   - davinci-002")
print("\n3. You need to request access to fine-tuning in your Azure subscription")
print("\nIf you're not in a supported region, consider using OpenAI directly instead of Azure OpenAI")
