import os
from typing import Optional
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatOllama  # Updated import for chat

from langchain_core.documents import Document
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
from pathlib import Path

# Load .env from the LangChainFundamental folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class NewsArticleSummarizer:
    def __init__(
        self,
        api_key: str = None,
        model_type: str = "azure",
        model_name: str = "gpt-4o-mini",
        azure_endpoint: str = None,
        azure_deployment: str = None,
        api_version: str = "2024-02-15-preview",
    ):
        """
        Initialize the summarizer with choice of model
        Args:
            api_key: API key (required for OpenAI/Azure models)
            model_type: 'openai', 'azure', or 'ollama'
            model_name: specific model name
            azure_endpoint: Azure OpenAI endpoint (required for Azure)
            azure_deployment: Azure deployment name (required for Azure)
            api_version: Azure API version (default: 2024-02-15-preview)
        """
        self.model_type = model_type
        self.model_name = model_name

        # Setup LLM based on model type
        if model_type == "azure":
            if not api_key:
                raise ValueError("API key is required for Azure OpenAI models")
            if not azure_endpoint:
                raise ValueError("Azure endpoint is required for Azure OpenAI")
            if not azure_deployment:
                raise ValueError("Azure deployment is required for Azure OpenAI")
            
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_key=api_key,
                api_version=api_version,
                #temperature=1,  # Using default temperature as 0 is not supported by this model
            )
        elif model_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_type == "ollama":
            # Using ChatOllama with proper configuration
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",  # Optional: for structured output
                timeout=120,  # Increased timeout for longer generations
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize text splitter for long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        """
        Fetch article content using newspaper3k
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """
        Create LangChain documents from text
        """
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        return docs

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        """
        Main summarization pipeline
        """
        # Fetch article
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        # Create documents
        docs = self.create_documents(article.text)

        # Define prompts based on summary type
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:  # concise summary
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        # Create prompts
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        # Create and run chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        # Generate summary
        summary = chain.invoke(docs)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main():
    # Example of using both models
    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-cambricons-first-profit-lands/"

    # Initialize Azure OpenAI summarizer
    azure_summarizer = NewsArticleSummarizer(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model_type="azure",
        model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    ollama_summarizer = NewsArticleSummarizer(
        model_type="ollama", model_name="llama3.2"  # or "llama3.2"
    )

    # Get summaries from both models
    print("\nGenerating Azure OpenAI Summary...")
    azure_summary = azure_summarizer.summarize(url, summary_type="detailed")

    print("\nGenerating Llama Summary...")
    llama_summary = ollama_summarizer.summarize(url, summary_type="detailed")

    # Print results
    for summary, model in [(azure_summary, "Azure OpenAI"), (llama_summary, "Llama")]:
        print(f"\n{model} Summary:")
        print("-" * 50)
        print(f"Title: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        # Print first document content
        print("\nFirst Document Content:")
        print(summary["summary"]["input_documents"][0].page_content)

        print("\Second Document Content:")
        print(summary["summary"]["input_documents"][1].page_content)


if __name__ == "__main__":
    main()