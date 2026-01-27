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
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant that performs unit conversions. "
                    "Show brief reasoning, then provide the final numeric result inside <answer> tags."
                )
            },
            # Shot 1: Integer conversion
            {"role": "user", "content": "How many meters are there in 6 km?"},
            {"role": "assistant", "content": "1 km = 1000 m, so 6 * 1000 = 6000. <answer>6000</answer>"},
            
            # Shot 2: Decimal/Precision handling
            {"role": "user", "content": "Convert 2.5 inches to centimeters."},
            {"role": "assistant", "content": "1 inch is 2.54 cm. 2.5 * 2.54 = 6.35. <answer>6.35</answer>"},
            
            # The actual question
            {"role": "user", "content": f"{question} Answer with <answer>...</answer>."}
        ]
    
        # apply_chat_template adds the necessary special tokens for SmolLM2
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def parse_answer(self, answer: str) -> float:
        if not answer or answer.isspace():
            return float("nan")
    
        # Pre-clean: Remove common unit noise that confuses floats
        clean_answer = answer.replace("$", "").strip()
    
        try:
            # 1. Standard Tag Match (Updated for Scientific Notation)
            m = re.search(r"<answer>\s*([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)", clean_answer, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", ""))

            # 2. Fraction Fallback
            m_frac = re.search(r"(\d+)\s*/\s*(\d+)", clean_answer)
            if m_frac:
                return float(m_frac.group(1)) / float(m_frac.group(2))

            # 3. Last Number Fallback (Updated for Scientific Notation)
            all_nums = re.findall(r"[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?", clean_answer)
            if all_nums:
                return float(all_nums[-1].replace(",", ""))

            return float("nan")
        except (ValueError, TypeError, ZeroDivisionError):
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
        return self.batched_generate([prompt])[0]

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

        # Generation params
        gen_kwargs = dict(
            max_new_tokens=256,
            min_new_tokens=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            num_beams=1,
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
        # Use a tiny bit of temperature to avoid the "empty string" trap
        generations = self.batched_generate(prompts, temperature=0.1) 
    
        results = []
        for g in generations:
            parsed = self.parse_answer(g)
            # print(f"DEBUG: Gen: '{g}' | Parsed: {parsed}") 
            results.append(parsed)
        return results


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
