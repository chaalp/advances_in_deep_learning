from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a precise unit conversion assistant. "
                    "Provide the final numeric result inside <answer> tags immediately. "
                    "Do not show reasoning." # Saves tokens and time
                ),
            },
            {"role": "user", "content": "How many meters are in 5.5 kilometers?"},
            {"role": "assistant", "content": "<answer>5500</answer>"},
            {"role": "user", "content": "Convert 12 inches to centimeters."},
            {"role": "assistant", "content": "<answer>30.48</answer>"},
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
