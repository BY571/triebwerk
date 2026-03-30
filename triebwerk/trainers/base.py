"""Base trainer with shared logic for model loading, generation, and training loop."""

import argparse
import json
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from triebwerk.compat import patch_amp_for_jetson, cast_model_to_fp16
from triebwerk.banner import print_banner, print_memory_map
from triebwerk.generation import generate_with_engine, generate_with_hf
from triebwerk.rewards import format_reward, correctness_reward, get_gsm8k_dataset
from triebwerk.utils import compute_advantages, compute_batch_token_logprobs
from triebwerk.lora_sync import LoRASyncer


class BaseTrainer(ABC):
    """Base class for GRPO-family trainers.

    Handles model loading, engine init, generation, training loop, logging,
    and checkpointing. Subclasses implement compute_loss() for the specific
    policy gradient variant.
    """

    def __init__(
        self,
        model="Qwen/Qwen3-0.6B",
        reward_funcs=None,
        loss_type="dapo",
        lora_rank=16,
        dry_run=False,
        lr=5e-6,
        epsilon=0.2,
        epsilon_high=None,
        temperature=1.0,
        top_p=0.9,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        mask_truncated=True,
        logging_steps=1,
        save_steps=100,
        empty_cache=None,
        output_dir="./checkpoints_engine",
        kv_bits=0,
        **kwargs,
    ):
        self.model_name = model
        self.reward_funcs = reward_funcs
        self.loss_type = loss_type
        self.lora_rank = lora_rank
        self.dry_run = dry_run
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.temperature = temperature
        self.top_p = top_p
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.gradient_checkpointing = gradient_checkpointing
        self.mask_truncated = mask_truncated
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.kv_bits = kv_bits
        self.extra_kwargs = kwargs

        # Auto-detect: empty_cache needed on Jetson unified memory, harmful on discrete GPUs
        if empty_cache is None:
            self.empty_cache = (
                torch.cuda.get_device_properties(0).is_integrated
                if torch.cuda.is_available() else False
            )
        else:
            self.empty_cache = empty_cache

        # Patch AMP for Jetson compatibility
        patch_amp_for_jetson()

        # Will be set during train()
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.syncer = None
        self.optimizer = None
        self.scaler = None
        self.device = None

    @property
    @abstractmethod
    def needs_reference_logprobs(self):
        """Whether this trainer needs old log-probs (True for GRPO/DAPO, False for DG)."""
        ...

    @abstractmethod
    def compute_loss(self, model, optimizer, scaler, samples):
        """Compute loss and do backward + optimizer step.

        Returns the average loss value (float).
        """
        ...

    def _build_args_namespace(self, max_steps, num_generations, max_completion_tokens):
        """Build an argparse.Namespace for banner/config compatibility."""
        return argparse.Namespace(
            model=self.model_name,
            max_steps=max_steps,
            num_generations=num_generations,
            max_completion_tokens=max_completion_tokens,
            lr=self.lr,
            lora_rank=self.lora_rank,
            loss_type=self.loss_type,
            output_dir=self.output_dir,
            dry_run=self.dry_run,
            epsilon=self.epsilon,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            top_p=self.top_p,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            gradient_checkpointing=self.gradient_checkpointing,
            mask_truncated=self.mask_truncated,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            empty_cache=self.empty_cache,
            kv_bits=self.kv_bits,
            _model_config=None,
            **{k: v for k, v in self.extra_kwargs.items()
               if isinstance(v, (int, float, str, bool, type(None)))},
        )

    def _load_engine(self, args):
        """Load C++ engine (before PyTorch to get clean VRAM)."""
        if self.dry_run:
            return

        from triebwerk.engine import load_engine
        self.engine, cfg = load_engine(
            args.num_generations, args.max_completion_tokens,
            self.temperature, self.top_p, kv_bits=self.kv_bits,
        )
        args._model_config = cfg

    def _load_model(self):
        """Load model with 4-bit quantization and LoRA."""
        print(f"\nLoading {self.model_name} (4-bit NF4, fp16)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cast_model_to_fp16(self.model)

        # LoRA
        lora_config = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  LoRA rank={self.lora_rank}, trainable: {trainable:,} / {total:,}")

        return trainable, total

    def _connect_engine(self, args):
        """Connect engine to PyTorch model (share embedding, init syncer)."""
        if self.engine is None:
            return

        # Limit PyTorch's CUDA allocator to prevent it from grabbing engine's memory
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5)

        # Share embedding (engine uses PyTorch's tensor, saves 311MB)
        embed_weight = self.model.base_model.model.model.embed_tokens.weight
        if embed_weight.dtype == torch.float16 and embed_weight.is_cuda:
            self.engine.share_embedding(embed_weight.data_ptr())

        self.syncer = LoRASyncer(
            self.model, self.engine,
            lora_alpha=self.lora_rank, lora_rank=self.lora_rank,
        )

        # Pre-allocate arena + capture CUDA graph (AFTER share_embedding so pointers are final)
        dummy_prompt = list(range(200))
        self.engine.generate_batch(
            [dummy_prompt] * args.num_generations,
            max_new_tokens=args.max_completion_tokens,
            temperature=self.temperature, top_p=self.top_p,
            eos_token_id=dummy_prompt[0],
        )
        print(f"  Arena pre-allocated + CUDA graph captured")

    def train(self, dataset=None, max_steps=300, num_generations=4,
              max_completion_tokens=512, stop_texts=None):
        """Run the full training loop.

        Args:
            dataset: HuggingFace dataset with "prompt" and "answer" columns.
                If None, loads GSM8K with default rewards.
            max_steps: Number of training steps.
            num_generations: Completions per prompt (G).
            max_completion_tokens: Max tokens per completion.
            stop_texts: List of strings to stop generation (e.g. ["</answer>"]).
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"{self.output_dir}/{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        args = self._build_args_namespace(max_steps, num_generations, max_completion_tokens)

        # 1. Load engine (before PyTorch for clean VRAM)
        self._load_engine(args)

        # 2. Banner
        print_banner(args, run_id)
        print("=" * 60)

        # 3. Load model
        trainable, total = self._load_model()

        # 4. Connect engine to model
        self._connect_engine(args)

        # 5. Memory map
        weights_path = (
            os.environ.get("ENGINE_WEIGHTS", "engine/weights_q4l")
            if self.engine else None
        )
        print_memory_map(self.engine, weights_path)

        # 6. Optimizer + scaler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.0)
        self.scaler = torch.amp.GradScaler("cuda")
        self.device = next(self.model.parameters()).device

        # 7. Dataset & rewards
        if dataset is not None:
            reward_funcs = self.reward_funcs
            stop_texts = stop_texts or []
        else:
            print("\nLoading GSM8K...")
            dataset = get_gsm8k_dataset()
            reward_funcs = self.reward_funcs or [format_reward, correctness_reward]
            stop_texts = stop_texts or ["</answer>"]
        dataset_iter = iter(dataset)

        stop_ids = []
        for text in stop_texts:
            stop_ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        warmup_steps = int(max_steps * self.warmup_ratio)

        # 8. Save config
        config_dict = {k: v for k, v in vars(args).items()
                       if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
        config_dict["run_id"] = run_id
        config_dict["trainable_params"] = trainable
        config_dict["total_params"] = total
        with open(f"{run_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # 9. Training loop
        all_metrics = []
        t_start = time.time()

        print(f"\nTraining: {max_steps} steps, G={num_generations}")
        print("=" * 60)

        for step in range(1, max_steps + 1):
            t_step = time.time()

            # LR schedule: linear warmup + cosine decay
            if step <= warmup_steps:
                lr = self.lr * step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Sample a prompt
            try:
                sample = next(dataset_iter)
            except StopIteration:
                dataset_iter = iter(dataset)
                sample = next(dataset_iter)

            prompt = sample["prompt"]
            answer = sample["answer"]

            # 1. Sync LoRA to engine
            if self.syncer is not None:
                self.syncer.sync()

            # 2. Generate G completions
            t_gen = time.time()
            if self.engine is not None:
                completions = generate_with_engine(
                    self.engine, self.tokenizer, prompt, num_generations,
                    max_completion_tokens, self.temperature, self.top_p, stop_ids,
                )
            else:
                completions = generate_with_hf(
                    self.model, self.tokenizer, prompt, num_generations,
                    max_completion_tokens, self.temperature, self.top_p, stop_ids,
                )
            t_gen = time.time() - t_gen

            # 3. Decode and score
            comp_texts = [
                self.tokenizer.decode(c["completion_ids"], skip_special_tokens=True)
                for c in completions
            ]
            expanded_answers = [answer] * num_generations

            rewards = torch.zeros(num_generations)
            for func in reward_funcs:
                scores = func(comp_texts, answer=expanded_answers)
                rewards += torch.tensor(scores, dtype=torch.float32)

            # 4. Compute advantages
            advantages = compute_advantages(rewards, num_generations)

            # 5. Compute reference log-probs (skip for trainers that don't need them)
            if self.needs_reference_logprobs:
                with torch.no_grad():
                    old_logprobs = compute_batch_token_logprobs(
                        self.model, completions, self.device,
                    )
            else:
                old_logprobs = [None] * len(completions)

            # 6. Build samples for the gradient step
            samples = []
            for i, c in enumerate(completions):
                mask_weight = 0.0 if (self.mask_truncated and c["truncated"]) else 1.0
                samples.append({
                    "prompt_ids": c["prompt_ids"],
                    "completion_ids": c["completion_ids"],
                    "old_logprobs": old_logprobs[i],
                    "advantage": advantages[i].item(),
                    "mask_weight": mask_weight,
                })

            # 7. Gradient step (subclass implements this)
            loss = self.compute_loss(self.model, self.optimizer, self.scaler, samples)

            # 8. Metrics
            total_tokens = sum(len(c["completion_ids"]) for c in completions)
            step_time = time.time() - t_step

            metrics = {
                "step": step,
                "loss": loss,
                "mean_reward": rewards.mean().item(),
                "max_reward": rewards.max().item(),
                "min_reward": rewards.min().item(),
                "reward_std": rewards.std().item(),
                "gen_time": t_gen,
                "gen_tokens": total_tokens,
                "gen_tok_per_s": total_tokens / t_gen if t_gen > 0 else 0,
                "step_time": step_time,
                "lr": lr,
                "n_truncated": sum(1 for c in completions if c["truncated"]),
            }
            all_metrics.append(metrics)

            if step % self.logging_steps == 0:
                elapsed = time.time() - t_start
                print(
                    f"[{step:>4}/{max_steps}] "
                    f"loss={loss:>7.4f}  "
                    f"reward={rewards.mean().item():>5.2f}  "
                    f"tok/s={metrics['gen_tok_per_s']:>5.0f}  "
                    f"gen={t_gen:>4.1f}s  "
                    f"step={step_time:>4.1f}s  "
                    f"lr={lr:.1e}  "
                    f"[{elapsed/60:.0f}m]"
                )

            # Save checkpoint
            if step % self.save_steps == 0:
                ckpt_path = f"{run_dir}/step_{step}_lora"
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                with open(f"{run_dir}/metrics.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)
                print(f"  Saved checkpoint: {ckpt_path}")

        # Final save
        total_time = time.time() - t_start
        final_path = f"{run_dir}/final_lora"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        summary = {
            "run_id": run_id,
            "total_time_s": total_time,
            "total_time_h": total_time / 3600,
            "steps": max_steps,
            "s_per_step": total_time / max_steps,
            "final_mean_reward": all_metrics[-1]["mean_reward"] if all_metrics else 0,
        }
        with open(f"{run_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        total_gen_tokens = sum(m["gen_tokens"] for m in all_metrics)
        total_gen_time = sum(m["gen_time"] for m in all_metrics)
        avg_tok_s = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0

        print(f"\n   \033[90m{'=' * 62}\033[0m")
        print(f"   \033[1m{'Training complete!':^62}\033[0m")
        print(f"   \033[90m{'-' * 62}\033[0m")
        print(f"   \033[1m Zeit:\033[0m       {total_time/3600:.1f}h ({total_time/max_steps:.1f}s/step)")
        print(f"   \033[1m Tokens:\033[0m     {total_gen_tokens:,} in {total_gen_time:.1f}s ({avg_tok_s:.0f} tok/s)")
        print(f"   \033[1m Belohnung:\033[0m  {summary['final_mean_reward']:.2f}")
        print(f"   \033[1m LoRA:\033[0m       {final_path}")
        print(f"   \033[1m Metriken:\033[0m   {run_dir}/metrics.json")
        print(f"   \033[90m{'=' * 62}\033[0m")

        return final_path
