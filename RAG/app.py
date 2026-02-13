import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "myData.csv")

def getText():
    query = input("Enter your query: ")
    return query

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    embedding = embeddings.embed_query("Hello world")
    print(f"Sample embedding for 'Hello world': {embedding[:5]}...")

    loader = CSVLoader(file_path=csv_path)
    data = loader.load()
    print(f"Successfully loaded {len(data)} documents from myData.csv")
    
    db = FAISS.from_documents(data, embeddings)
    print("FAISS vector database created successfully!")
    
except FileNotFoundError as e:
    print(f"Error loading myData.csv: File not found - {e}")
except Exception as e:
    print(f"Error loading myData.csv: {str(e)}")

while False:
    query = getText()
    if query.lower() in ['exit', 'quit', 'q']:
        print("Exiting the application.")
        break
    try:
        docs = db.similarity_search(query)
        print("Top similar documents:")
        for i, doc in enumerate(docs[:2]):
            print(f"Document {i+1}:\n{doc.page_content}\n")
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")