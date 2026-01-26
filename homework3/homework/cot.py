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
                    "You solve unit conversion problems. "
                    "Be concise. "
                    "Think briefly, then output ONLY the final numeric answer inside "
                    "<answer>...</answer>. "
                    "Do not include units or extra text."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Example:\n"
                    "Question: How many minutes are in 2 hours?\n"
                    "Reasoning: 2 hours = 2 * 60 = 120 minutes.\n"
                    "<answer>120</answer>"
                ),
            },
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
