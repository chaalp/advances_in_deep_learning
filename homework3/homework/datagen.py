import json
from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    # Use the 1.7B model for higher quality reasoning rollouts
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = CoTModel(checkpoint=checkpoint) 
    
    dataset = Dataset("train")
    rft_data = []

    questions = [item[0] for item in dataset]
    # Generate diverse reasoning paths
    results = model.batched_generate(
        questions, 
        num_return_sequences=oversample, 
        temperature=temperature
    )

    for i, attempts in enumerate(results):
        question, correct_val = dataset[i]
        for attempt in attempts:
            predicted_val = model.parse_answer(attempt)
            if is_answer_valid(predicted_val, correct_val):
                # Save Question, Rounded Answer, and the Full Reasoning Path
                rft_data.append([question, round(float(correct_val), 3), attempt])
                break 

    with open(output_json, "w") as f:
        json.dump(rft_data, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
