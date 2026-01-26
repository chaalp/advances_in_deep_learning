from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    #question_len = len(tokenizer(question)["input_ids"])
    question_len = len(tokenizer(question + " ")["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    # sanity check must supervise at least 1 token
    if all(x == -100 for x in labels):
        raise ValueError("All labels are -100 — supervision is empty. Check question_len / max_length.")

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    #raise NotImplementedError()

    # Ensure numeric, then round for stability
    try:
        ans = float(answer)
    except (TypeError, ValueError):
        # Fallback: leave as-is if something unexpected comes in
        return {"question": prompt, "answer": f"<answer>{answer}</answer>"}

    # Use compact formatting: keep up to 3 decimals, strip trailing zeros
    ans_str = f"{ans:.3f}".rstrip("0").rstrip(".")
    return {
        "question": prompt + " Answer with <answer>...</answer>.",
        "answer": f"<answer>{ans_str}</answer>",
    }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    #raise NotImplementedError()
    #test_model(output_dir)

    """
    Train a LoRA adapter for SmolLM2 on the unit-conversion dataset.

    Expected usage (from repo root):
        python -m homework.sft train --output_dir homework/sft_model
    """
    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator
    from peft import LoraConfig, get_peft_model

    # Hyperparams (overridable via **kwargs)
    learning_rate = float(kwargs.get("learning_rate", 2e-4))
    num_train_epochs = float(kwargs.get("num_train_epochs", 5))
    per_device_train_batch_size = int(kwargs.get("per_device_train_batch_size", 32))
    gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(kwargs.get("warmup_ratio", 0.03))
    weight_decay = float(kwargs.get("weight_decay", 0.0))
    logging_steps = int(kwargs.get("logging_steps", 25))
    save_strategy = kwargs.get("save_strategy", "epoch")

    # LoRA size control (< ~20MB recommended in README)
    lora_r = int(kwargs.get("lora_r", 8))
    lora_alpha = int(kwargs.get("lora_alpha", 32))
    lora_dropout = float(kwargs.get("lora_dropout", 0.05))

    # Data
    trainset = Dataset("train")
    llm = BaseLLM()

    train_dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    # LoRA
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        #target_modules="all-linear",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.print_trainable_parameters()

    llm.model.config.use_cache = False
    
    # Fix for gradient_checkpointing + LoRA on GPU (recommended by README)
    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_checkpointing=True,
        #fp16=(llm.device == "cuda"),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        remove_unused_columns=False,  # important since our dataset returns dicts
        label_names=["labels"], # Common default
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=llm.model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save ONLY the adapter (PeftModel)
    trainer.model.save_pretrained(output_dir)

    test_model(output_dir)

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
