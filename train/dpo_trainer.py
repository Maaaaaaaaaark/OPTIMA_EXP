"""Single-GPU LoRA DPO trainer used independently for Alice and Bob."""
import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_prompt_length", type=int, default=1280)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--attn_implementation", default="eager")
    return parser.parse_args()


def main():
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    dataset = load_from_disk(args.dataset_path)
    train = dataset["train"]
    if len(train) == 0:
        raise RuntimeError(f"DPO train split is empty: {args.dataset_path}")
    eval_dataset = dataset["test"] if len(dataset["test"]) else None
    print(f"[dpo] {args.output_dir}: {len(train)} train / {len(dataset['test'])} test pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
    except (ImportError, ValueError) as exc:
        print(f"[dpo] attention fallback after: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype
        )
    model.config.use_cache = False

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                item.strip() for item in args.lora_target_modules.split(",") if item.strip()
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        save_strategy="no",
        evaluation_strategy="no",
        remove_unused_columns=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # With PEFT and ref_model=None, TRL evaluates the frozen reference by
    # disabling the adapter.  This avoids holding a second 2B model on T4.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=train,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        peft_config=peft_config,
    )
    trainer.train()
    if args.use_lora:
        merged = trainer.model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.model.config.use_cache = True
        trainer.save_model(args.output_dir)
    print(f"[dpo] done: {args.output_dir}")


if __name__ == "__main__":
    main()
