import ollama

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": user_input}]
    )

    print("Assistant:", response["message"]["content"])
