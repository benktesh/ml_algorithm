from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Define a prompt template
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# Check environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

print(f"Endpoint: {endpoint}")
print(f"API Key: {'***' if api_key else 'Not set'}")
print(f"Deployment: {deployment}")

if not deployment:
    print("\nERROR: AZURE_OPENAI_DEPLOYMENT_NAME is not set in your .env file!")
    print("Please add: AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name")
    exit(1)

# Use AzureChatOpenAI
model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    deployment_name=deployment,
    api_version=api_version,
    #temperature=0.7,
)

# Chain the prompt, model, and output parser
chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({"topic": "bears"})
print(response)