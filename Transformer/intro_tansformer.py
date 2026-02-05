from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

print("Transformers library imported successfully.")

def create_simple_llm():
    model_name = "distilgpt2"
    generator = pipeline("text-generation", model=model_name, tokenizer=AutoTokenizer.from_pretrained(model_name))
    return generator

def generate_text(generator, prompt, max_length=100):
    
    result = generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7, truncation=True)
    return result[0]['generated_text']

def run_llm_demo():
    """
    Demonstrates basic LLM functionality with explanations
    """
    print("🤖 Loading Simple LLM Model...")
    generator = create_simple_llm()

    print("\n✨ Simple LLM Demo ✨")
    print("This demo shows basic text generation using a small language model")

    # Example prompts to demonstrate different capabilities
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python programming is",
    ]

    for prompt in prompts:
        print("\n🔹 Prompt:", prompt)
        print("🔸 Generated:", generate_text(generator, prompt))
        input("\nPress Enter to see next example...")

def explain_process():
    """
    Explains the LLM process with a simple example
    """
    print("\n🎓 How it works:")
    print("1. Input text → Tokenization → Numbers")
    print("2. Numbers → Model Processing → Prediction")
    print("3. Prediction → New Token → Output Text")

    # Simple tokenization example
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("\n📝 Example Tokenization:")
    print(f"Original text: '{text}'")
    print(f"As tokens (numbers): {tokens}")
    print(f"Decoded back: '{decoded}'")

generator = create_simple_llm()

def interactive_demo():
    """
    Allows users to interact with the model
    """
    generator = create_simple_llm()

    print("\n🤖 Interactive LLM Demo")
    print("Type your prompts (or 'quit' to exit)")

    while True:
        prompt = input("\n✍️ Enter your prompt: ")
        if prompt.lower() == "quit":
            break

        response = generate_text(generator, prompt)
        print("\n💭 Generated response:")
        print(response)
        
if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Run basic demonstration")
    print("2. Interactive mode")
    print("3. Explain the process")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("Invalid choice!")