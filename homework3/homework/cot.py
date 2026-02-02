from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        #messages = [
        #    {"role": "system", "content": "You are a helpful assistant. Be concise. Always reason step-by-step and provide the final answer inside <answer></answer> tags."},
        #    {"role": "user", "content": "How many feet are in 2 yards?"},
        #    {"role": "assistant", "content": "1 yard is 3 feet. So, 2 yards * 3 = 6 feet. <answer>6</answer>"},
        #    {"role": "user", "content": question}
        #]

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a precise unit conversion assistant. "
                    "First, identify the conversion factor. Second, perform the calculation. "
                    "Finally, provide the numeric result inside <answer> tags. "
                    "Respond with reasoning then the answer."
                ),
            },
            # Few-Shot Example 1: Standard conversion
            {"role": "user", "content": "How many meters are in 5.5 kilometers?"},
            {"role": "assistant", "content": "1 kilometer is 1000 meters. So, 5.5 * 1000 = 5500. <answer>5500</answer>"},
            
            # Few-Shot Example 2: Precision/Decimal handling
            {"role": "user", "content": "Convert 12 inches to centimeters."},
            {"role": "assistant", "content": "1 inch is 2.54 cm. 12 * 2.54 = 30.48. <answer>30.48</answer>"},
            
            # The actual question
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def load(checkpoint: str = None) -> CoTModel:
    model = CoTModel(checkpoint=checkpoint)
    model.model_name = "cot"
    return model 

def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
