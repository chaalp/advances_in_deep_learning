import json
from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = CoTModel(checkpoint=checkpoint) # This must be the CoTModel from Part 2
    
    dataset = Dataset("train")
    rft_data = []

    # Format questions into Chat Prompts
    # This uses your Part 2 implementation of format_prompt
    prompts = [model.format_prompt(item[0]) for item in dataset]

    # Generate results (Temperature 0.8 is better for diversity in RFT)
    results = model.batched_generate(
        prompts, 
        num_return_sequences=oversample, 
        temperature=temperature
    )

    for i, attempts in enumerate(results):
        question, correct_val = dataset[i]
        for attempt in attempts:
            predicted_val = model.parse_answer(attempt)
            if is_answer_valid(predicted_val, correct_val) and "</answer>" in attempt:
                # Save the Reasoning path (the full 'attempt' string)
                rft_data.append([question, round(float(correct_val), 3), attempt])
                break 

    with open(output_json, "w") as f:
        json.dump(rft_data, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
