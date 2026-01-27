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
    # This MUST be an exact copy of the messages list in BaseLLM.format_prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that performs unit conversions. Show brief reasoning, then provide the final numeric result inside <answer> tags."},
        {"role": "user", "content": "How many meters are there in 6 km?"},
        {"role": "assistant", "content": "1 km = 1000 m, so 6 * 1000 = 6000. <answer>6000</answer>"},
        {"role": "user", "content": "Convert 2.5 inches to centimeters."},
        {"role": "assistant", "content": "1 inch is 2.54 cm. 2.5 * 2.54 = 6.35. <answer>6.35</answer>"},
        {"role": "user", "content": f"{question} Answer with <answer>...</answer>."},
        {"role": "assistant", "content": answer} 
    ]
    
    # Apply template and tokenize
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=256) # Increased length
    
    # Calculate prompt length for masking
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    labels = full["input_ids"].copy()
    for i in range(len(labels)):
        if i < prompt_len or full["attention_mask"][i] == 0:
            labels[i] = -100 # Mask everything except the final answer
            
    full["labels"] = labels
    return full

def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    #raise NotImplementedError()

    # 1. Clean number formatting
    try:
        ans_str = f"{float(answer):g}"
    except:
        ans_str = str(answer)

    # 2. Return raw components
    # Do NOT add instruction strings; let tokenize/format_prompt handle it
    return {
        "question": prompt,
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
    learning_rate = float(kwargs.get("learning_rate", 5e-5))
    num_train_epochs = float(kwargs.get("num_train_epochs", 5))
    per_device_train_batch_size = int(kwargs.get("per_device_train_batch_size", 32))
    gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(kwargs.get("warmup_ratio", 0.03))
    weight_decay = float(kwargs.get("weight_decay", 0.0))
    logging_steps = int(kwargs.get("logging_steps", 25))
    save_strategy = kwargs.get("save_strategy", "epoch")

    # LoRA params
    lora_r = int(kwargs.get("lora_r", 16))
    lora_alpha = int(kwargs.get("lora_alpha", 32))
    lora_dropout = float(kwargs.get("lora_dropout", 0.05))

    trainset = Dataset("train")
    llm = BaseLLM()

    train_dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 1. Initialize LoRA
    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.print_trainable_parameters()

    # 2. Fix for gradient checkpointing + LoRA
    #if llm.device == "cuda":
    llm.model.enable_input_require_grads()

    # 3. Ensure the model is in training mode
    llm.model.train()
    
    # 4. Explicitly ensure LoRA weights are trainable
    for name, param in llm.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # 5. Disable cache to save memory and avoid warnings during training
    llm.model.config.use_cache = False

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=1e-4,              # Slightly higher for small model
        num_train_epochs=3,              # 3-5 is usually sufficient
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,               # 5% warmup
        weight_decay=0.01,               # Added for generalization
        gradient_checkpointing=True,     # Disable if VRAM allows for speed
        bf16=torch.cuda.is_bf16_supported(), # Auto-detect BF16
        fp16=not torch.cuda.is_bf16_supported(), # Fallback to FP16
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,     # Keep for custom dataset
        label_names=["labels"],
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
