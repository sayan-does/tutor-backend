import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2 import Qwen2Tokenizer

def main():
    # Load tokenizer and model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tutor initialization prompt
    tutor_context = "You are a patient, knowledgeable tutor helping a student learn. Provide clear, supportive explanations. Break down complex topics, ask guiding questions, and offer encouragement."

    print("AI Tutor Chatbot. Type 'exit' to quit.")
    print("What subject would you like to study today?")

    while True:
        # Get user input
        user_input = input("Student: ")

        # Check for exit condition
        if user_input.lower() == 'exit':
            print("Tutor: Great learning session! Keep up the good work!")
            break

        # Prepare full prompt with tutor context
        full_prompt = f"{tutor_context}\n\nStudent: {user_input}\nTutor:"

        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        # Generate response
        outputs = model.generate(
            **inputs, 
            max_length=200,  # Increased length for more detailed explanation
            do_sample=True,  # Enable more creative responses
            temperature=0.7,  # Balanced creativity and coherence
            top_p=0.9
        )

        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the tutor's response
        tutor_response = response.split("Tutor:")[-1].strip()

        # Print response
        print("Tutor:", tutor_response)

if __name__ == "__main__":
    main()