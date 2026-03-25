"""GRPO training using the C++ inference engine for generation.

52.7 tok/s generation (7.3x over HF baseline) + PyTorch backward (3.1s).
The engine handles: prefill, decode, GPU sampling — all in C++/CUDA.
PyTorch handles: forward pass for log-probs, GRPO loss, backward, optimizer.

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -e CUDA_HOME=/usr/local/cuda-12.6 \\
      -e PATH=/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \\
      -e TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas \\
      grpo-jetson bash -c 'PYTHONPATH=engine/build python3 train_gsm8k_engine.py --max-steps 300'
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "engine/build")
import jetson_engine

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def compute_rewards(completions, answer):
    format_rewards, correctness_rewards = [], []
    for comp in completions:
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", comp, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", comp, re.DOTALL))
        format_rewards.append(1.0 if has_reasoning and has_answer else (0.5 if has_answer else 0.0))
        extracted = extract_xml_answer(comp)
        correctness_rewards.append(2.0 if extracted == answer else -1.0)
    combined = [f + c for f, c in zip(format_rewards, correctness_rewards)]
    return combined, format_rewards, correctness_rewards


def get_gsm8k_prompts():
    data = load_dataset("openai/gsm8k", "main")["train"]
    prompts, answers = [], []
    for item in data:
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ])
        answers.append(extract_hash_answer(item["answer"]))
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training — C++ Engine (52.7 tok/s)")
    print(f"Run: {run_id}")
    print("=" * 60)

    # ── Load C++ engine for generation ──
    print("\nLoading C++ engine...")
    engine = jetson_engine.Engine(1024)
    engine.load_weights("engine/weights")
    print("  Engine ready")

    # ── Load PyTorch model for backward pass ──
    print(f"\nLoading PyTorch model for backward ({args.model})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cast_model_to_fp16(model)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    # ── Stop tokens for engine ──
    stop_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    # ── Dataset ──
    print("\nLoading GSM8K...")
    prompts, answers = get_gsm8k_prompts()
    print(f"  {len(prompts)} examples")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # ── Save config ──
    config = {
        "run_id": run_id, "model": args.model, "max_steps": args.max_steps,
        "num_generations": args.num_generations, "max_completion_tokens": args.max_completion_tokens,
        "lr": args.lr, "lora_rank": args.lora_rank, "temperature": args.temperature,
        "generation": "cpp_engine_52tok/s",
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining: {args.max_steps} steps, G={args.num_generations}")
    print(f"  Generation: C++ engine (52.7 tok/s)")
    print(f"  Backward: PyTorch (4-bit + LoRA)")
    print("=" * 60)

    all_metrics = []
    t_start = time.time()

    for step in range(args.max_steps):
        idx = step % len(prompts)
        prompt_msgs = prompts[idx]
        answer = answers[idx]

        # ── Generate with C++ engine (fast) ──
        model.eval()
        torch.cuda.synchronize()
        t_gen = time.perf_counter()

        input_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(input_text).input_ids

        completions = []
        for g in range(args.num_generations):
            engine.reset()
            token_ids = engine.generate(
                input_ids,
                max_new_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                top_p=0.9,
                eos_token_id=eos_id,
                stop_token_ids=stop_ids,
            )
            completions.append(tokenizer.decode(token_ids, skip_special_tokens=True))

        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen

        # ── Rewards ──
        combined, fmt_rewards, cor_rewards = compute_rewards(completions, answer)

        # ── GRPO advantages ──
        rewards_t = torch.tensor(combined, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        advantages = (rewards_t - mean_r) / std_r if std_r > 1e-8 else rewards_t - mean_r

        # ── Backward (PyTorch) ──
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t_train = time.perf_counter()

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        prompt_len = len(tokenizer(prompt_text).input_ids)

        total_loss = 0.0
        n_valid = 0

        for comp_text, adv in zip(completions, advantages):
            full_msgs = prompt_msgs + [{"role": "assistant", "content": comp_text}]
            full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)
            tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                               max_length=1024).to(device)

            if tokens["input_ids"].shape[1] <= prompt_len + 1:
                del tokens
                continue

            outputs = model(**tokens)
            logits = outputs.logits[:, :-1, :]
            labels = tokens["input_ids"][:, 1:]
            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), reduction="none",
            ).reshape(labels.shape)

            mask = torch.zeros_like(labels, dtype=torch.float32)
            if prompt_len - 1 < mask.shape[1]:
                mask[:, prompt_len - 1:] = 1.0

            mean_log_prob = (log_probs * mask).sum() / mask.sum().clamp(min=1)
            loss = -adv.item() * mean_log_prob
            loss.backward()
            total_loss += loss.item()
            n_valid += 1
            del outputs, logits, labels, log_probs, mask, mean_log_prob, loss, tokens

        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t_train = time.perf_counter() - t_train

        total_tokens = sum(len(tokenizer(c).input_ids) for c in completions)
        metrics = {
            "step": step + 1, "gen_time": t_gen, "train_time": t_train,
            "total_time": t_gen + t_train,
            "mean_reward": mean_r.item(),
            "mean_format": sum(fmt_rewards) / len(fmt_rewards),
            "mean_correctness": sum(cor_rewards) / len(cor_rewards),
            "loss": total_loss / max(n_valid, 1), "n_valid": n_valid,
            "gen_tok_s": total_tokens / max(t_gen, 0.01),
        }
        all_metrics.append(metrics)

        elapsed = time.time() - t_start
        eta = (elapsed / (step + 1)) * (args.max_steps - step - 1)
        print(f"  Step {step+1}/{args.max_steps}: "
              f"reward={metrics['mean_reward']:.2f} "
              f"(fmt={metrics['mean_format']:.2f}, cor={metrics['mean_correctness']:.2f}) "
              f"loss={metrics['loss']:.4f} "
              f"gen={t_gen:.1f}s train={t_train:.1f}s "
              f"[{metrics['gen_tok_s']:.0f} tok/s] "
              f"ETA {eta/3600:.1f}h")

        if (step + 1) % args.save_every == 0:
            ckpt_dir = f"{run_dir}/checkpoint-{step+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # ── Final save ──
    total_time = time.time() - t_start
    lora_path = f"{run_dir}/final_lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    summary = {
        "run_id": run_id, "total_time_s": total_time, "total_time_h": total_time / 3600,
        "steps": args.max_steps, "s_per_step": total_time / args.max_steps,
        "avg_gen_time": sum(m["gen_time"] for m in all_metrics) / len(all_metrics),
        "avg_train_time": sum(m["train_time"] for m in all_metrics) / len(all_metrics),
        "avg_reward": sum(m["mean_reward"] for m in all_metrics[-50:]) / min(50, len(all_metrics)),
        "generation": "cpp_engine",
    }
    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump({"steps": all_metrics}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Time: {total_time/3600:.1f}h ({total_time/args.max_steps:.1f}s/step)")
    print(f"  LoRA: {lora_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
