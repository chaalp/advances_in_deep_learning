import json
from .cot import load
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    # Load the CoT model (consider using 1.7B for better results)
    model = load()
    dataset = Dataset("train")
    rft_data = []

    # Process questions in batches for speed
    questions = [item[0] for item in dataset]
    
    # Generate multiple diverse reasoning paths
    results = model.batched_generate(
        questions, 
        num_return_sequences=oversample, 
        temperature=temperature
    )

    for i, attempts in enumerate(results):
        question, correct_val = dataset[i]
        
        for attempt in attempts:
            predicted_val = model.parse_answer(attempt)
            
            # Check if the generated reasoning leads to the correct answer
            if is_answer_valid(predicted_val, correct_val): 
                # Rounding the correct_val here for the JSON entry
                # ensures consistent training labels
                rounded_correct = round(float(correct_val), 3)
                
                # We store the 'attempt' because it contains the 
                # reasoning steps + the <answer> tag
                rft_data.append([question, rounded_correct, attempt])
                break # Only need one successful path per question

    with open(output_json, "w") as f:
        json.dump(rft_data, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
