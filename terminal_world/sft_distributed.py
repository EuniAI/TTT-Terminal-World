"""DDP lifecycle and token-weighted SFT updates, isolated from TTT."""
from contextlib import contextmanager
from datetime import timedelta
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from terminal_world.model.meta_model import ensure_finite
from terminal_world.model.sft import batch_loss


class DistributedContext:
    def __init__(self, device, rank=0, world_size=1):
        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size

    @property
    def primary(self):
        return self.rank == 0

    def root_call(self, callback):
        """All ranks participate; rank zero's I/O errors reach every worker."""
        result, error = None, [None]
        if self.primary:
            try:
                result = callback()
            except Exception as exc:
                if self.world_size == 1:
                    raise
                error[0] = f"{type(exc).__name__}: {exc}"
        if self.world_size > 1:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError("rank zero operation failed: " + error[0])
        return result

    def any_stop(self, requested):
        value = torch.tensor(int(bool(requested)), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def rng_state(self):
        return {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None}

    def restore_rng(self, checkpoint):
        states = checkpoint.get("extra", {}).get("sft_distributed_rng")
        if states and len(states) == self.world_size:
            state = states[self.rank]
            cuda = state["cuda"]
        else:
            # Legacy single-process checkpoints or a changed worker count.
            # The model has no dropout; data ordering remains the global order.
            state = checkpoint["rng"]
            cuda_states = state.get("cuda", [])
            cuda = cuda_states[min(self.rank, len(cuda_states) - 1)] if cuda_states else None
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if self.device.type == "cuda" and cuda is not None:
            torch.cuda.set_rng_state(cuda, self.device)


@contextmanager
def distributed_context(args):
    size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0")) if size > 1 else 0
    device = torch.device(args.device)
    if size > 1 and device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    args.device = str(device)
    initialized = False
    try:
        if size > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=10))
            initialized = True
        yield DistributedContext(device, rank, size)
    finally:
        if initialized:
            dist.destroy_process_group()


class SFTLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, steps, pad_id):
        return batch_loss(self.model, steps, pad_id=pad_id)


def wrap_model(model, context):
    if context.world_size == 1:
        return None
    ids = [context.device.index] if context.device.type == "cuda" else None
    return DistributedDataParallel(SFTLoss(model), device_ids=ids, broadcast_buffers=False)


def train_distributed_batch(model, ddp, optimizer, state, batch, *, pad_id, context):
    model.validate_fast(state.fast_weights)
    optimizer.zero_grad(set_to_none=True)
    n = len(batch.steps)
    start, end = n * context.rank // context.world_size, n * (context.rank + 1) // context.world_size
    local = batch.steps[start:end]
    global_tokens = sum(int((s.loss_masks & s.attention_mask).sum()) for s in batch.steps)
    local_tokens = sum(int((s.loss_masks & s.attention_mask).sum()) for s in local)
    # An uneven final batch can leave a worker empty. It still participates in
    # the same DDP backward, with exactly zero loss/gradient contribution.
    local_loss, records = ddp(local or batch.steps[:1], pad_id)
    weighted = local_loss * (context.world_size * local_tokens / global_tokens)
    weighted.backward()
    ensure_finite([p.grad for p in model.parameters()], "distributed SFT gradients")
    optimizer.step()
    ensure_finite(model.parameters(), "distributed SFT parameters")
    for values in optimizer.state.values():
        ensure_finite([v for v in values.values() if torch.is_tensor(v)], "optimizer state")
    groups = context.gather(records if local else [])
    records = [record for group in groups for record in group]
    state.next_cursor, state.is_open = batch.next_cursor, batch.is_open
    state.windows_completed += 1
    return sum(r["nll_sum"] for r in records) / global_tokens, records
