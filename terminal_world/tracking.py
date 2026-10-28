"""Optional scalar-only W&B tracking; training checkpoints own the run identity."""
import math
from pathlib import Path
import uuid
import warnings


def scalar_metrics(data, prefix="train"):
    result = {}
    for key, value in data.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    return result


class Tracker:
    def __init__(self, run=None, identity=None):
        self.run = run
        self.identity = identity

    @property
    def enabled(self):
        return self.run is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            try:
                self.run.finish(exit_code=0 if exc_type is None else 1)
            except Exception:
                if exc_type is None:
                    raise
                warnings.warn("W&B finish failed while handling a training error", RuntimeWarning)

    def log(self, metrics):
        if self.run is not None:
            # Custom x axis survives resumes/segments without relying on W&B's internal _step.
            self.run.log(scalar_metrics(metrics))


def start_tracker(args, config, state, model_info, manifest_digest, previous=None):
    project = getattr(args, "wandb_project", None)
    mode = getattr(args, "wandb_mode", None) or ("online" if project else "disabled")
    if mode == "disabled":
        return Tracker(identity=previous)  # Preserve identity even during untracked continuation.
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B requested; install with: python -m pip install -e '.[tracking]'") from exc

    fork = getattr(args, "wandb_new_run", False)
    entity = getattr(args, "wandb_entity", None)
    project = project or (previous or {}).get("project") or "terminal-world"
    entity = entity or (previous or {}).get("entity")
    if previous and not fork:
        if project != previous["project"] or entity != previous.get("entity"):
            raise ValueError("W&B project/entity changed; use --wandb-new-run to fork logging")
    # Offline SDK runs cannot resume history. Give every offline segment its own ID.
    resume = bool(previous and previous.get("mode") == "online" and mode == "online" and not fork)
    run_id = previous["id"] if resume else uuid.uuid4().hex[:16]
    directory = Path(getattr(args, "wandb_dir", None) or Path(args.output).parent)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**config.to_dict(), "model_info": model_info, "manifest_digest": manifest_digest}
    if previous and not resume:
        payload["parent_run_id"] = previous["id"]
    # init must not consume the trainer's Python/Torch RNG streams.
    import random
    import torch
    python_rng = random.getstate()
    try:
        with torch.random.fork_rng():
            run = wandb.init(
                project=project, entity=entity, id=run_id,
                name=getattr(args, "wandb_name", None), mode=mode,
                resume="must" if resume else ("never" if mode == "online" else None),
                dir=str(directory), config=payload, save_code=False,
                settings=wandb.Settings(console="off"),
            )
    finally:
        random.setstate(python_rng)
    try:
        run.define_metric("train/windows")
        run.define_metric("train/*", step_metric="train/windows")
        run.summary["last_session_start_window"] = state.windows_completed
    except BaseException:
        run.finish(exit_code=1)
        raise
    identity = {"id": run.id, "project": run.project, "entity": run.entity, "mode": mode}
    return Tracker(run, identity)
