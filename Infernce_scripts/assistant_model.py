"""
Tesrting the use of assistant model for Faster Infernce Showed Good results but slower sometime
"""


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
# Give hugginface token to access model acc


# Configuration
model_name = "mistralai/Mistral-7B-v0.1"
assistant_checkpoint = "openaccess-ai-collective/mistral-100m-textbooks"
input_length = 128
output_length = 128
total_tokens = input_length + output_length
throughput_target = 200  # tokens/sec
device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quant_config ,torch_dtype=torch.float16, attn_implementation="sdpa",device_map="cuda")
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint,torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_outputs(inputs, output_length=128):
    #TODO DEBUG the use od sdp_kernel in this NOT WORKING
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs,assistant_model=assistant_model, max_new_tokens=output_length, pad_token_id=tokenizer.eos_token_id)
    return outputs

user_input = input("Enter your prompt: ")
input_text = [user_input]  #TODO Concurrency = 32
inputs = tokenizer(input_text, return_tensors="pt")

output_length = 128
input_tokens = len(tokenizer.encode(user_input))
total_tokens = input_tokens + output_length

# Benchmarking
start_time = time.time()

outputs = generate_outputs(inputs, output_length)

end_time = time.time()
total_time = end_time - start_time

throughput = total_tokens/ total_time

print(f"Total time taken: {total_time:.2f} seconds")
print(f"Total tokens processed: {total_tokens}")
print(f"Throughput: {throughput:.2f} tokens/sec")

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")


