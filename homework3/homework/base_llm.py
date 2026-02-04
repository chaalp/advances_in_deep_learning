from typing import overload

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        if checkpoint is None:
            checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            torch_dtype=torch.bfloat16 # or torch.float16
        ).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question
        
    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
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
    ) -> list[str] | list[list[str]]:
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
        #from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.

        micro_batch_size = 16

        #if len(prompts) > micro_batch_size:
        #    return [
        #        r
        #        for idx in tqdm(
        #            range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
        #        )
        #        for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
        #    ]
        
        if len(prompts) > micro_batch_size:
            out = []
            for i in range(0, len(prompts), micro_batch_size):
                out.extend(self.batched_generate(prompts[i:i+micro_batch_size], num_return_sequences, temperature))
            return out

         #raise NotImplementedError()

        torch.cuda.empty_cache()

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        is_reasoning_model = getattr(self, "model_name", "") in ["cot", "rft"]
        max_tokens = 128 if is_reasoning_model else 48

        # Call model.generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            num_return_sequences=num_return_sequences or 1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Slice to get only new tokens
        generated_ids = output_ids[:, input_length:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if num_return_sequences is not None:
            # Reshape into list of lists: [len(prompts), num_return_sequences]
            return [decoded[i : i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
        
        return decoded


    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
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
