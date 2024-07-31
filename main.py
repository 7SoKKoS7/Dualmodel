import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Отключение предупреждений
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Проверка наличия CUDA
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please ensure that you have a compatible GPU and CUDA installed.")

model_name = "EleutherAI/gpt-j-6b"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)

# Перемещение модели на GPU
model = model.to("cuda")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        response = generate_response(user_input)
        print(f"Bot: {response}")
