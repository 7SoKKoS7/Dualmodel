
# README for Dual-Model Chatbot System

---

### Project Overview

This project aims to develop an efficient dual-model chatbot system that integrates a locally hosted GPT-J 6B model for data processing and an API-based model for generating high-quality responses. The local model handles information retrieval and parsing, while the API-based model utilizes the retrieved data to generate contextually accurate responses.

### Table of Contents

1. [Introduction](#introduction)
2. [Effectiveness of Dual-Model Systems](#effectiveness-of-dual-model-systems)
3. [Project Structure](#project-structure)
4. [Installation and Setup](#installation-and-setup)
5. [Local Model Configuration](#local-model-configuration)
6. [Chat Context Management](#chat-context-management)
7. [API Integration](#api-integration)
8. [Testing and Deployment](#testing-and-deployment)
9. [References](#references)

---

### Introduction

The dual-model chatbot system leverages both local and cloud resources to balance performance and accuracy. This approach is particularly beneficial for applications requiring robust data handling and sophisticated natural language processing.

### Effectiveness of Dual-Model Systems

Research has shown that combining local and cloud-based models can significantly enhance the efficiency and scalability of chatbot systems. Local models can efficiently manage real-time data processing, while cloud-based models offer superior language generation capabilities due to their extensive training on diverse datasets.

- **Scalability:** Dual-model systems are scalable and can handle increasing workloads by distributing tasks between local and cloud resources.
- **Performance:** Local models can quickly preprocess data, reducing latency for the cloud model to generate responses.
- **Accuracy:** Cloud-based models, such as those provided by OpenAI, offer high accuracy in language generation due to their advanced architectures and large training datasets.

For detailed insights, refer to studies on [chatbot architectures and scalability](https://link.springer.com/article/10.1007/s00354-023-00145-3) and [optimizing chatbot effectiveness](https://www.mdpi.com/2076-3417/10/9/3214).

### Project Structure

1. **Installation and Setup:** Steps to install necessary libraries and set up the environment.
2. **Local Model Configuration:** Instructions to configure and test the local model for data processing.
3. **Chat Context Management:** Methods to save and manage chat history to maintain context.
4. **API Integration:** Steps to integrate the local model with the API-based model for generating responses.
5. **Testing and Deployment:** Guidelines for testing the system and deploying it in a production environment.

### Installation and Setup

#### Prerequisites

- Python 3.12
- Virtual environment setup (venv or conda)
- Necessary hardware: NVIDIA RTX 4060 Ti, 64 GB RAM

#### Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **.env File:**
   Create a `.env` file in the root directory to store your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SEARCH_API_KEY=your_search_api_key
   ```

### Local Model Configuration

1. **Load and Test the Local Model (GPT-J 6B):**
   ```python
   import torch
   from transformers import GPTJForCausalLM, AutoTokenizer

   model_name = "EleutherAI/gpt-j-6B"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = GPTJForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)

   def generate_text(prompt):
       inputs = tokenizer(prompt, return_tensors="pt").to(device)
       outputs = model.generate(inputs.input_ids, max_length=100)
       return tokenizer.decode(outputs[0], skip_special_tokens=True)

   print(generate_text("Hello! How are you?"))
   ```

### Chat Context Management

1. **Save Chat History:**
   ```python
   chat_history = []

   def add_to_chat_history(user_input, model_response):
       chat_history.append({"user": user_input, "model": model_response})
   ```

2. **Retrieve and Use Context:**
   ```python
   def get_context():
       return " ".join([f"User: {entry['user']} Model: {entry['model']}" for entry in chat_history])
   ```

### API Integration

1. **Set Up OpenAI API:**
   ```python
   import openai
   from dotenv import load_dotenv
   import os

   load_dotenv()
   openai.api_key = os.getenv('OPENAI_API_KEY')

   def generate_text_with_openai(prompt):
       response = openai.Completion.create(
           engine="gpt-4",
           prompt=prompt,
           max_tokens=150
       )
       return response.choices[0].text.strip()

   context = get_context()
   response = generate_text_with_openai(f"{context} User: What is the weather today?")
   print(response)
   ```

### Testing and Deployment

1. **Testing:**
   - Ensure all components work together by running integrated tests.
   - Validate the system with sample conversations and evaluate performance.

2. **Deployment:**
   - Deploy the system on a server or cloud platform.
   - Monitor performance and optimize as needed.

### requirements.txt

```
torch
transformers
requests
beautifulsoup4
openai
python-dotenv
```

### References

- [Architectural Scalability of Conversational Chatbot: The Case of ChatGPT](https://link.springer.com/article/10.1007/s00354-023-00145-3)
- [Modern Chatbot Systems: A Technical Review](https://link.springer.com/article/10.1007/s00354-023-00146-0)
- [Optimizing Chatbot Effectiveness through Advanced Syntactic Analysis](https://www.mdpi.com/2076-3417/10/9/3214)
- [Large-Language-Models (LLM)-Based AI Chatbots: Architecture, In-Depth Analysis, and Their Performance Evaluation](https://link.springer.com/article/10.1007/s00354-023-00147-7)

---

This README provides a structured approach to building a dual-model chatbot system, ensuring a balance between local data processing and advanced language generation capabilities. For further information, refer to the linked studies and documentation.
