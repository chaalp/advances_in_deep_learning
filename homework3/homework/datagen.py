def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()

    import json
    from pathlib import Path

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    assert oversample >= 2, "oversample should be >= 2 for meaningful rejection sampling"
    assert temperature > 0, "temperature should be > 0 for diverse rollouts"

    trainset = Dataset("train")
    model = CoTModel()

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[list] = []

    # Generate in batches for speed/memory safety
    batch_size = 16
    for start in range(0, len(trainset), batch_size):
        batch = [trainset[i] for i in range(start, min(len(trainset), start + batch_size))]
        questions = [q for (q, _) in batch]
        golds = [float(a) for (_, a) in batch]

        prompts = [model.format_prompt(q) for q in questions]

        # batched_generate returns list[list[str]] when num_return_sequences is int
        generations = model.batched_generate(
            prompts,
            num_return_sequences=int(oversample),
            temperature=float(temperature),
        )

        # For each question, pick the first correct completion
        for q, gold, gens in zip(questions, golds, generations):
            chosen = None
            for g in gens:
                pred = model.parse_answer(g)
                if pred == pred and is_answer_valid(pred, gold):  # pred==pred filters NaN
                    chosen = g.strip()
                    break

            if chosen is not None:
                results.append([q, gold, chosen])

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} / {len(trainset)} samples to {out_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
