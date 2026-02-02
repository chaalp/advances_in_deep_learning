import json
from .cot import load
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    model = load()
    dataset = Dataset("train")
    rft_data = []

    questions = [item[0] for item in dataset]
    # Generate multiple paths for each question
    results = model.batched_generate(questions, num_return_sequences=oversample, temperature=temperature)

    for i, attempts in enumerate(results):
        question, correct_val = dataset[i]
        for attempt in attempts:
            predicted_val = model.parse_answer(attempt)
            if is_answer_valid(predicted_val, correct_val):
                # Save the reasoning + the answer format
                rft_data.append([question, correct_val, attempt])
                break # Move to next question once one correct path is found

    with open(output_json, "w") as f:
        json.dump(rft_data, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
