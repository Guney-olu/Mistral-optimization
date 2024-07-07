import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import torch
import time

login(token=os.environ.get("XYZ")) # ADD your token here

def mode_setup(model_name):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config,
                                                torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

def generate_outputs(inputs,model,tokenizer,output_length=128):
    #using sdp_kernel for fast inference 
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
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--prompt", required=True, help="Input prompt for the model")
    parser.add_argument("--output_length", type=int, default=128, help="Length of the output text")

    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    model,tokenizer = mode_setup(args.model)
    input_text = [args.prompt]
    inputs = tokenizer(input_text, return_tensors="pt")

    # Parameters
    output_length = args.output_length
    input_tokens = len(tokenizer.encode(args.prompt))
    total_tokens = input_tokens + output_length

    # Benchmarking
    start_time = time.time()
    outputs = generate_outputs(inputs,model,tokenizer,output_length)
    end_time = time.time()
    total_time = end_time - start_time
    # Calculating throughput
    throughput = total_tokens / total_time

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/sec")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
