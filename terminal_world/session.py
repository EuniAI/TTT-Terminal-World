import torch
from terminal_world.dataloader.events import transition
from terminal_world.model.data import Event
from terminal_world.model.meta_model import detach_fast, ensure_finite, inner_loop_step, step_loss


class Session:
    def __init__(self, model, fast_weights, encoder, *, alpha, clip_norm=None, is_open=False):
        model.validate_fast(fast_weights)
        self.model, self.fast_weights, self.encoder = model, fast_weights, encoder
        self.alpha, self.clip_norm, self.is_open = alpha, clip_norm, is_open
        self.observations = 0
        # Session owns its model. Evaluation must copy before creating a Session.
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def generate_tokens(self, command):
        event = Event("shell", command)
        transition(self.is_open, event)
        prompt = self.encoder.prompt(event)
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt], device=device)
        generated = []
        cache = None  # Never reuse across events or fast-weight updates.
        for _ in range(self.encoder.max_output_tokens):
            logits, cache = self.model.forward_cached(ids, self.fast_weights, cache)
            logits = logits[:, -1]
            ensure_finite([logits], "generation logits")
            # Protocol markers other than OBS_END cannot be generated as output text.
            forbidden = [v for k, v in self.encoder.tokenizer.ids.items() if k != "obs_end"]
            logits[:, forbidden] = -torch.inf
            token = int(logits.argmax(-1))
            generated.append(token)
            if token == self.encoder.tokenizer.ids["obs_end"]:
                break
            ids = torch.tensor([[token]], device=device)
        return generated

    def predict(self, command):
        tokens = self.generate_tokens(command)
        end = self.encoder.tokenizer.ids["obs_end"]
        return self.encoder.tokenizer.decode([t for t in tokens if t != end])

    @torch.enable_grad()
    def apply_event(self, event, *, adapt=True):
        next_open = transition(self.is_open, event)
        step = self.encoder.encode(event).to(next(self.model.parameters()).device)
        nll, n = step_loss(self.model, self.fast_weights, step)
        ensure_finite([nll], "session loss")
        stats = {"update_norm": 0.0, "clipped": False}
        if adapt:
            new_fast, stats = inner_loop_step(self.fast_weights, nll / n, alpha=self.alpha,
                                              create_graph=False, clip_norm=self.clip_norm, return_stats=True)
            self.fast_weights = detach_fast(new_fast)
        self.is_open = next_open
        self.observations += 1
        return {"kind": event.kind, "nll_sum": float(nll.detach()), "tokens": n,
                **step.metadata, **stats}

    def observe(self, command, real_output):
        return self.apply_event(Event("shell", command, real_output))

    def control(self, command):
        return self.apply_event(Event("control", command))
