"""Standalone full-parameter SFT trainer for the Qwen OPTIMA pipeline.

No alignment-handbook, no Ray, no DeepSpeed: plain ``python train/sft_trainer.py``
on a single GPU. Completion-only masking is done with trl's
DataCollatorForCompletionOnlyLM on the native Qwen2.5 chat template, so the
model trains on assistant turns only (partner turns stay 'user').

The dataset is pre-templated (dataset_text_field="text", packing=False);
rows longer than max_seq_length must be filtered when the dataset is built
(train/dataset_build.py) because SFTTrainer truncates from the tail.
"""
import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="SFT trainer (Alice or Bob)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--response_template", type=str, default="<|im_start|>assistant\n")
    parser.add_argument("--instruction_template", type=str, default="<|im_start|>user\n")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

    dataset = load_from_disk(args.dataset_path)["train"]
    print(f"[sft] {args.output_dir}: {len(dataset)} training rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = (
        torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
        )
    except (ImportError, ValueError) as e:
        print(f"[sft] attn_implementation={args.attn_implementation} failed ({e}); "
              "falling back to default attention")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=model_dtype,
        )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                name.strip() for name in args.lora_target_modules.split(",")
                if name.strip()
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        max_grad_norm=args.max_grad_norm,
        save_strategy="no",
        save_total_limit=None,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        instruction_template=args.instruction_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()
    # Save a normal full checkpoint that the next vLLM iteration can serve.
    # PEFT's default save is adapter-only, so merge LoRA into the base first.
    if args.use_lora:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    print(f"[sft] done: {args.output_dir}")


if __name__ == "__main__":
    main()
