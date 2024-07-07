import torch
import time

class Benchmark:
    def __init__(self, model, tokenizer, concurrency, output_length):
        self.model = model
        self.tokenizer = tokenizer
        self.concurrency = concurrency
        self.output_length = output_length

    def generate_outputs(self, inputs):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            inputs = inputs.to('cuda')
            outputs = self.model.generate(
                **inputs, do_sample=True, max_new_tokens=self.output_length, pad_token_id=self.tokenizer.eos_token_id)
        return outputs

    def benchmark(self, user_input):
        input_text = [user_input] * self.concurrency
        inputs = self.tokenizer(input_text, return_tensors="pt")

        input_tokens = len(self.tokenizer.encode(user_input))
        total_tokens = input_tokens + self.output_length

        start_time = time.time()
        outputs = self.generate_outputs(inputs)
        end_time = time.time()
        total_time = end_time - start_time

        throughput = total_tokens * self.concurrency / total_time

        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Total tokens processed: {total_tokens * self.concurrency}")
        print(f"Throughput: {throughput:.2f} tokens/sec")

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
