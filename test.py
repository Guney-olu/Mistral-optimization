from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.insert(0, 'REPO PATH') #TODO Create modules instead for easy import
from Bechmarks.Base_benchmark import Benchmark
import torch
import time

# Configuration
model_name = "mistralai/Mistral-7B-v0.1" # model of you choice 

quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quant_config ,torch_dtype=torch.float16, attn_implementation="sdpa",device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

benchmark = Benchmark(model=model, tokenizer=tokenizer, concurrency=64, output_length=128)
user_input = input("Enter your prompt: ")
#user_input = "I am a cake"
print(benchmark.benchmark(user_input))