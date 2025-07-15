from manager import PromptManager

def test_prompt_system():
    # Initialize the prompt manager
    manager = PromptManager()
    
    # Test the education summarize template
    try:
        result = manager.render(
            "education_summarize",
            context="Plants need water and sunlight to grow.",
            remaining_time=45,
            language="Hindi"
        )
        print("Education Summarize Template:")
        print(result)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error with education_summarize template: {e}")
    
    # Test the translation template
    try:
        result = manager.render(
            "translate",
            text="Hello, how are you?",
            target_language="Telugu"
        )
        print("Translation Template:")
        print(result)
    except Exception as e:
        print(f"Error with translate template: {e}")

if __name__ == "__main__":
    test_prompt_system() 