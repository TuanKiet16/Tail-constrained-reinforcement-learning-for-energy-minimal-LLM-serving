from __future__ import annotations

import time

_PASSAGE = (
    "Large language models answer questions by predicting one token after another. "
    "Each new token requires a pass through every layer of the network, and the work "
    "is done by a graphics processor that draws more power the faster it is clocked. "
    "A data centre that serves many users at once has to decide how quickly each "
    "answer must appear, how many requests can share the processor, and how much "
    "electricity it is willing to spend. When few people are asking, the processor "
    "can slow down without anyone noticing. When a burst of requests arrives, it must "
    "speed up again or the queue will grow and people will wait. Choosing the right "
    "speed at every moment is a control problem, and reinforcement learning is one way "
    "to solve it: an agent tries different settings, observes the delay and the energy "
    "that follow, and gradually prefers the settings that keep answers prompt while "
    "using as little energy as possible. "
)


def _tiny_random_model():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=4096, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
    )
    pool = torch.randint(5, config.vocab_size, (8192,), generator=torch.Generator().manual_seed(1))
    return Qwen2ForCausalLM(config), pool


class HFEngine:
    def __init__(self, model_id: str, dtype: str = "float16", device: str | None = None, record: bool = False):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        if model_id == "tiny-random":
            model, pool = _tiny_random_model()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
            ids = tokenizer(_PASSAGE * 8, add_special_tokens=False)["input_ids"]
            pool = torch.tensor(ids, dtype=torch.long)
        self.model = model.to(self.device).eval()
        self.pool = pool
        self.record = record  # keep generated tokens (tests only)
        self.outputs: dict[int, list[int]] = {}
        self._clear()

    
    def _clear(self) -> None:
        self.cache = None  
        self.mask = None 
        self.pos = None 
        self.last = None  
        self.rids: list[int] = []

    def reset(self) -> None:
        self._clear()
        if self.device != "cpu":
            self.torch.cuda.empty_cache()

    @property
    def batch(self) -> int:
        return 0 if self.last is None else int(self.last.shape[0])

    def _sync(self) -> None:
        if self.device != "cpu":
            self.torch.cuda.synchronize()

    def _check(self, best_logits) -> None:
        if not bool(self.torch.isfinite(best_logits).all()):
            raise RuntimeError(f"{self.model_id} produced non-finite logits; try dtype: bfloat16 or float32 in the config")

    def prompt_ids(self, rid: int, length: int):
        start = (rid * 7919) % len(self.pool)
        idx = (self.torch.arange(length) + start) % len(self.pool)
        return self.pool[idx]

    
    def prefill(self, requests) -> None:
        torch = self.torch
        with torch.inference_mode():
            n, length = len(requests), max(r.prompt_len for r in requests)
            ids = torch.zeros((n, length), dtype=torch.long)
            mask = torch.zeros((n, length), dtype=torch.long)
            for i, r in enumerate(requests):
                ids[i, length - r.prompt_len:] = self.prompt_ids(r.rid, r.prompt_len)
                mask[i, length - r.prompt_len:] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            positions = (mask.cumsum(-1) - 1).clamp_(min=0)
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
            best, last = out.logits[:, -1, :].max(-1)
            self._check(best)
            pos = mask.sum(-1)
            if not hasattr(out.past_key_values, "layers"):
                raise RuntimeError("This engine needs transformers 5 or newer: pip install -U transformers")
            if any(type(layer).__name__ != "DynamicLayer" for layer in out.past_key_values.layers):
                raise NotImplementedError(
                    f"{self.model_id} uses a sliding-window key/value cache, which this engine cannot batch. "
                    "Pick another model, for example one of the Qwen2.5 sizes."
                )
            if self.cache is None:
                self.cache, self.mask, self.pos, self.last = out.past_key_values, mask, pos, last
            else:
                self._join(out.past_key_values, mask, pos, last)
            self.rids += [r.rid for r in requests]
            if self.record:
                for r, tok in zip(requests, last.tolist()):
                    self.outputs[r.rid] = [tok]
            self._sync()

    def _join(self, cache, mask, pos, last) -> None:
        torch, pad = self.torch, self.torch.nn.functional.pad
        grow_old = max(0, mask.shape[1] - self.mask.shape[1])
        grow_new = max(0, self.mask.shape[1] - mask.shape[1])
        for mine, theirs in zip(self.cache.layers, cache.layers):
            # pad the sequence dimension (second to last) on the left
            mine.keys = torch.cat([pad(mine.keys, (0, 0, grow_old, 0)), pad(theirs.keys, (0, 0, grow_new, 0))], 0)
            mine.values = torch.cat([pad(mine.values, (0, 0, grow_old, 0)), pad(theirs.values, (0, 0, grow_new, 0))], 0)
        self.mask = torch.cat([pad(self.mask, (grow_old, 0)), pad(mask, (grow_new, 0))], 0)
        self.pos = torch.cat([self.pos, pos], 0)
        self.last = torch.cat([self.last, last], 0)

    def decode(self, max_steps: int, until: float) -> tuple[int, float]:
        torch = self.torch
        start, steps = time.perf_counter(), 0
        with torch.inference_mode():
            while steps < max_steps:
                self.mask = torch.cat([self.mask, torch.ones_like(self.mask[:, :1])], 1)
                out = self.model(
                    input_ids=self.last[:, None], attention_mask=self.mask,
                    position_ids=self.pos[:, None], past_key_values=self.cache, use_cache=True,
                )
                self.cache = out.past_key_values
                best, self.last = out.logits[:, -1, :].max(-1)
                self.pos = self.pos + 1
                if self.record:
                    for rid, tok in zip(self.rids, self.last.tolist()):
                        self.outputs[rid].append(tok)
                self._sync()  # the token is "delivered" here, as it would be when streaming
                steps += 1
                if time.perf_counter() >= until:
                    break
            elapsed = time.perf_counter() - start
            if steps:
                self._check(best)
        return steps, elapsed

    def remove(self, rows: list[int]) -> None:
        torch = self.torch
        gone = set(rows)
        keep = [i for i in range(self.batch) if i not in gone]
        if not keep:
            self._clear()
            return
        with torch.inference_mode():
            index = torch.tensor(keep, device=self.last.device)
            mask = self.mask.index_select(0, index)
            lead = int((mask.sum(0) > 0).int().argmax())  # leading columns nobody uses any more
            for layer in self.cache.layers:
                layer.keys = layer.keys.index_select(0, index)[:, :, lead:, :]
                layer.values = layer.values.index_select(0, index)[:, :, lead:, :]
            self.mask = mask[:, lead:]
            self.pos, self.last = self.pos.index_select(0, index), self.last.index_select(0, index)
        self.rids = [self.rids[i] for i in keep]
