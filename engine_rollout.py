"""Custom rollout function for TRL's GRPOTrainer using C++ engine.

The rollout function:
1. Generates completions using the C++ engine (fast, 57 tok/s)
2. Computes log-probs using PyTorch forward pass (needed for GRPO ratio)
3. Returns in TRL's expected format: {prompt_ids, completion_ids, logprobs}

TRL then handles the correct GRPO loss (clipped surrogate, advantage normalization).
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "engine/build")
import jetson_engine


class EngineRollout:
    """Wraps C++ engine generation + PyTorch log-prob computation for TRL."""

    def __init__(self, engine, pytorch_model, tokenizer,
                 max_completion_tokens=512, temperature=1.0, top_p=0.9,
                 lora_syncer=None):
        self.engine = engine
        self.model = pytorch_model
        self.tokenizer = tokenizer
        self.max_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.lora_syncer = lora_syncer

        # Pre-compute stop tokens
        self.stop_ids = tokenizer.encode("</answer>", add_special_tokens=False)
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, prompts, trainer):
        """TRL rollout_func interface: (prompts, trainer) -> dict.

        Args:
            prompts: list of prompts (each is either a string or list of message dicts)
            trainer: GRPOTrainer instance (has num_generations, etc.)

        Returns:
            dict with keys: prompt_ids, completion_ids, logprobs
            Each is a list with len = len(prompts) * num_generations
        """
        num_generations = trainer.num_generations if hasattr(trainer, 'num_generations') else 4
        device = next(self.model.parameters()).device

        # Sync LoRA weights to engine before generation
        if self.lora_syncer:
            self.lora_syncer.sync()

        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []

        for prompt in prompts:
            # Tokenize prompt
            if isinstance(prompt, list):
                # Conversational format
                input_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
            else:
                input_text = prompt

            prompt_ids = self.tokenizer(input_text).input_ids

            for g in range(num_generations):
                # Generate with C++ engine
                self.engine.reset()
                completion_token_ids = self.engine.generate(
                    prompt_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    eos_token_id=self.eos_id,
                    stop_token_ids=self.stop_ids,
                )

                # Compute log-probs via PyTorch forward pass
                # Full sequence = prompt + completion
                full_ids = prompt_ids + completion_token_ids
                input_tensor = torch.tensor([full_ids], device=device)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits[0, :-1, :]  # (seq_len-1, vocab)
                    targets = input_tensor[0, 1:]         # (seq_len-1,)

                    # Per-token log-probs
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

                    # Only keep completion log-probs (skip prompt)
                    completion_log_probs = token_log_probs[len(prompt_ids) - 1:]

                all_prompt_ids.append(prompt_ids)
                all_completion_ids.append(completion_token_ids)
                all_logprobs.append(completion_log_probs.cpu())

                del outputs, logits, log_probs, token_log_probs, input_tensor

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
        }
