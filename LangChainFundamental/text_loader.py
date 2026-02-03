from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import pprint
import re
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


#Load the document
documents = TextLoader("./LangChainFundamental/doc/dream.txt").load()
#print(document[:10])


# Split the text into smaller chunks first
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

#clean the text after splitting
cleaned_texts = [clean_text(text.page_content) for text in texts]

#print(cleaned_texts)

# Use free local embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

retriever = FAISS.from_texts(cleaned_texts, embedding=embeddings).as_retriever(search_kwargs={"k": 3} )

query = "who is benktesh"
docs = retriever.invoke(query)


pprint.pprint(f" => DOCS: {docs}:")

#chat with model and our docs

prompt = ChatPromptTemplate.from_template("Please use the folllowing {docs},and answer the following questions")


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
response = chain.invoke({"docs": docs, "query": query})
print(response)