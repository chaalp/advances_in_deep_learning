import json
import torch
from .cot import load
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    # Use 1.7B model for rollouts
    checkpoint_17b = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = load(checkpoint=checkpoint_17b)
    
    dataset = Dataset("train")
    rft_data = []

    # Small micro-batch size to prevent OOM with the 1.7B model
    micro_batch_size = 8 

    for i in range(0, len(dataset), micro_batch_size):
        batch = dataset.data[i : i + micro_batch_size]
        questions = [item[0] for item in batch]
        correct_vals = [item[1] for item in batch]
        
        # Use Chat/CoT formatting
        prompts = [model.format_prompt(q) for q in questions]

        # Inference only: Disable gradient calculation to save VRAM
        with torch.no_grad():
            results = model.batched_generate(
                prompts, 
                num_return_sequences=oversample, 
                temperature=temperature
            )

        for j, attempts in enumerate(results):
            question, correct_val = questions[j], correct_vals[j]
            for attempt in attempts:
                predicted_val = model.parse_answer(attempt)
                # Filter for correct reasoning paths using 5% tolerance
                if is_answer_valid(predicted_val, correct_val):
                    rft_data.append([question, round(float(correct_val), 3), attempt])
                    break 

        # Clear fragmentation after each micro-batch
        torch.cuda.empty_cache()

    with open(output_json, "w") as f:
        json.dump(rft_data, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
