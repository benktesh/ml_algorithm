from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_DEPLOYMENT_NAME = os.environ["AZURE_DEPLOYMENT_NAME"]


def create_chat_model():
    """Create and configure Azure OpenAI chat model"""
    model = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        # GPT-5 Nano only supports default temperature (1)
    )
    return model


def prompt_template_example():
    """Using LangChain Prompt Templates"""
    print("Using Prompt Templates with LCEL")
    
    model = create_chat_model()
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    
    # Create chain using LCEL (LangChain Expression Language)
    chain = prompt | model | StrOutputParser()
    
    # Execute chain
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?"
    })
    
    print(f"Translation: {result}\n")
    return result


def simple_chain_example():
    """Using LangChain Chains"""
    print("Using Simple Chain")
    
    model = create_chat_model()
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Tell me a {adjective} joke about {topic}"
    )
    
    # Build chain with LCEL
    chain = prompt | model | StrOutputParser()
    
    # Execute
    result = chain.invoke({"adjective": "funny", "topic": "programming"})
    print(f"Joke: {result}\n")
    return result


def conversation_with_history_example():
    """Multi-turn conversation using message history"""
    print("Using Message History")
    
    model = create_chat_model()
    
    # Manually manage conversation history
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Hi, my name is Alice and I love Python programming."),
    ]
    
    # First response
    response = model.invoke(messages)
    print(f"Response 1: {response.content}\n")
    
    # Add to history
    messages.append(response)
    
    # Second interaction - should remember the name
    messages.append(HumanMessage(content="What's my name and what do I love?"))
    response = model.invoke(messages)
    print(f"Response 2: {response.content}\n")
    
    return messages


def sequential_chain_example():
    """Chain multiple operations together"""
    print("Using Sequential Chain with LCEL")
    
    model = create_chat_model()
    
    # First chain: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a random interesting topic related to {field}"
    )
    topic_chain = topic_prompt | model | StrOutputParser()
    
    # Second chain: Write about the topic
    article_prompt = ChatPromptTemplate.from_template(
        "Write a brief 2-sentence summary about: {topic}"
    )
    article_chain = article_prompt | model | StrOutputParser()
    
    # Combine chains using LCEL
    full_chain = (
        {"topic": topic_chain}
        | RunnablePassthrough.assign(article=article_chain)
    )
    
    result = full_chain.invoke({"field": "artificial intelligence"})
    print(f"Generated topic: {result['topic']}")
    print(f"Article: {result['article']}\n")
    return result


def chat_with_history_example():
    """Chat with explicit message history using LCEL"""
    print("Chat with Explicit History (LCEL)")
    
    model = create_chat_model()
    
    # Create prompt with message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions concisely."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | model | StrOutputParser()
    
    # Maintain chat history
    history = []
    
    # First message
    response1 = chain.invoke({"history": history, "input": "My favorite color is blue."})
    print(f"Assistant: {response1}")
    
    # Update history
    history.extend([
        HumanMessage(content="My favorite color is blue."),
        AIMessage(content=response1)
    ])
    
    # Second message
    response2 = chain.invoke({"history": history, "input": "What's my favorite color?"})
    print(f"Assistant: {response2}\n")
    
    return history


def streaming_chain_example():
    """Streaming with chains"""
    print("Streaming Chain Example")
    
    model = create_chat_model()
    
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {topic}"
    )
    
    chain = prompt | model | StrOutputParser()
    
    print("Haiku: ", end="")
    for chunk in chain.stream({"topic": "machine learning"}):
        print(chunk, end="", flush=True)
    print("\n")


def batch_processing_example():
    """Process multiple inputs in batch"""
    print("Batch Processing Example")
    
    model = create_chat_model()
    
    prompt = ChatPromptTemplate.from_template(
        "What is the capital of {country}?"
    )
    
    chain = prompt | model | StrOutputParser()
    
    # Process multiple inputs
    inputs = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"}
    ]
    
    results = chain.batch(inputs)
    
    for country, result in zip([i["country"] for i in inputs], results):
        print(f"{country}: {result}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("LangChain Framework Examples with Azure OpenAI")
    print("=" * 70)
    print()
    
    # Example 1: Prompt Templates
    print("1. " + "=" * 66)
    prompt_template_example()
    
    # Example 2: Simple Chain
    print("2. " + "=" * 66)
    simple_chain_example()
    
    # Example 3: Conversation with History
    print("3. " + "=" * 66)
    conversation_with_history_example()
    
    # Example 4: Sequential Chain
    print("4. " + "=" * 66)
    sequential_chain_example()
    
    # Example 5: Chat with History
    print("5. " + "=" * 66)
    chat_with_history_example()
    
    # Example 6: Streaming
    print("6. " + "=" * 66)
    streaming_chain_example()
    
    # Example 7: Batch Processing
    print("7. " + "=" * 66)
    batch_processing_example()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
