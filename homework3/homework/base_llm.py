from typing import overload

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        # Define the instruction as a system message to guide the model's behavior
        # Determine if we should use Chain of Thought based on the class attribute
        is_cot = getattr(self, "model_name", "") == "cot"

        if is_cot:
            system_content = (
                "You are a precise unit conversion assistant. "
                "First, identify the conversion factor. Second, perform the calculation. "
                "Finally, provide the numeric result inside <answer> tags. "
                "Respond with reasoning then the answer."
            )
            assistant_shot = "1 kilometer is 1000 meters. So, 5.5 * 1000 = 5500. <answer>5500</answer>"
        else:
            system_content = (
                "You are a unit converter. Provide the numeric result inside <answer> tags immediately. "
                "Do not show reasoning."
            )
            assistant_shot = "<answer>6000</answer>"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "How many meters are in 6 km?" if not is_cot else "How many meters are in 5.5 km?"},
            {"role": "assistant", "content": assistant_shot},
            {"role": "user", "content": f"{question} Answer with <answer>...</answer>."}
        ]
    
        # apply_chat_template adds the necessary special tokens for SmolLM2
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def parse_answer(self, answer: str) -> float:
        # 1. Try to extract from the <answer> tag first
        # 2. Fallback to the last numerical value in the string
        tag_match = re.search(r"<answer>\s*([-+]?[\d,]*\.?\d+)", answer)
    
        # If no tag, find all numbers and pick the last one
        if tag_match:
            raw_val = tag_match.group(1)
        else:
            numbers = re.findall(r"[-+]?[\d,]*\.?\d+", answer)
            if not numbers:
                return float("nan")
            raw_val = numbers[-1]

        try:
            val = float(raw_val.replace(",", ""))
            # Return as int if it's a whole number, else float
            return int(val) if val.is_integer() else val
        except ValueError:
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        #return self.batched_generate([prompt])[0]
        # 'prompt' here is actually the raw question from the benchmark
        formatted_prompt = self.format_prompt(prompt) 
        return self.batched_generate([formatted_prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ):
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            out = []
            for i in range(0, len(prompts), micro_batch_size):
                out.extend(self.batched_generate(prompts[i:i+micro_batch_size], num_return_sequences, temperature))
            return out
        
        #raise NotImplementedError()

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

         # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Detect which model variant is running
        is_cot = getattr(self, "model_name", "") == "cot"
    
        # CoT needs more tokens for reasoning (128)
        # SFT needs fewer tokens (48) to pass the 40s timeout in non-batch mode
        max_tokens = 128 if is_cot else 48

        # Generation params
        gen_kwargs = dict(
            max_new_tokens=max_tokens,     # Reduced for speed/timeout safety
            min_new_tokens=1,      # Forces the model to generate a response
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            use_cache=True,
        )

        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = int(num_return_sequences)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # Decode ONLY the newly generated tokens (mask out prompt)
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # If we generated multiple sequences per prompt, reshape to list[list[str]]
        if num_return_sequences is not None:
            n, k = len(prompts), num_return_sequences
            return [decoded[i*k:(i+1)*k] for i in range(n)]

        return decoded
    
    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        
        # FORCE temperature 0 and ensure no sampling
        generations = self.batched_generate(prompts, temperature=0) 
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
