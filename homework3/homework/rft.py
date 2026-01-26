from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()

    import json
    from pathlib import Path

    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator
    from peft import LoraConfig, get_peft_model

    from .data import DATA_DIR
    from .sft import TokenizedDataset, tokenize

    # Load RFT dataset
    rft_path = Path(kwargs.get("rft_path", DATA_DIR / "rft.json"))
    with rft_path.open() as f:
        rft_data = json.load(f)

    class RFTDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # entry: [question, correct_float, reasoning_with_<answer>]
            q, _, reasoning = self.data[idx]
            return (q, reasoning)

    def format_rft_example(prompt: str, reasoning: str) -> dict[str, str]:
        # Here "answer" field is actually reasoning+answer text, supervised fully after the question
        return {"question": prompt + " Answer with <answer>...</answer>.", "answer": reasoning}


    trainset = RFTDataset(rft_data)

    llm = BaseLLM()

    train_dataset = TokenizedDataset(llm.tokenizer, trainset, format_rft_example)

    # LoRA params
    lora_r = int(kwargs.get("lora_r", 16))
    lora_alpha = int(kwargs.get("lora_alpha", 64))
    lora_dropout = float(kwargs.get("lora_dropout", 0.05))

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

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Training hyperparams
    learning_rate = float(kwargs.get("learning_rate", 2e-4))
    num_train_epochs = float(kwargs.get("num_train_epochs", 3))
    per_device_train_batch_size = int(kwargs.get("per_device_train_batch_size", 32))
    gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(kwargs.get("warmup_ratio", 0.03))
    logging_steps = int(kwargs.get("logging_steps", 25))
    save_strategy = kwargs.get("save_strategy", "epoch")

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=True,
        #fp16=(llm.device == "cuda"),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"], # Common default
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=llm.model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)

    # Reuse the same evaluator from SFT
    test_model(output_dir)

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
