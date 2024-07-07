import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import torch
import time
from peft import PeftModel




def mode_setup(model_name, adapter_name=None):
    if adapter_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
        model = PeftModel.from_pretrained(model, adapter_name)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.bos_token_id = 1
        stop_token_ids = [0]
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quant_config,
            torch_dtype=torch.float16, 
            attn_implementation="sdpa", 
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_outputs(inputs, model, tokenizer, output_length=128):
    # Using sdp_kernel for fast inference 
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        inputs = inputs.to('cuda')
        outputs = model.generate(
            **inputs, do_sample=True, max_new_tokens=output_length, pad_token_id=tokenizer.eos_token_id)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Inference Run TEST",
        description="Running Mistral chat",
        epilog="Help"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--adapter_name", help="Adapter name")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--prompt", required=True, help="Input prompt for the model")
    parser.add_argument("--output_length", type=int, default=128, help="Length of the output text")
    parser.add_argument("--hf_token",help="hugging face token")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
    
    access_token = args.hf_token
    login(token = access_token)

    model, tokenizer = mode_setup(args.model, args.adapter_name)
    input_text = [args.prompt]
    inputs = tokenizer(input_text, return_tensors="pt")

    # Parameters
    output_length = args.output_length
    input_tokens = len(tokenizer.encode(args.prompt))
    total_tokens = input_tokens + output_length

    # Benchmarking
    start_time = time.time()
    outputs = generate_outputs(inputs, model, tokenizer, output_length)
    end_time = time.time()
    total_time = end_time - start_time
    # Calculating throughput
    throughput = total_tokens / total_time

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/sec")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
