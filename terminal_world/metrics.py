"""Detached, token-weighted metrics. No training graphs are retained."""


def summarize(records):
    result = {}
    for kind in ("shell", "control"):
        group = [r for r in records if r["kind"] == kind]
        count = sum(r["tokens"] for r in group)
        result[kind] = {"events": len(group), "tokens": count,
                        "nll": sum(r["nll_sum"] for r in group) / count if count else None,
                        "update_norm_mean": sum(r.get("update_norm", 0) for r in group) / len(group) if group else None,
                        "clip_rate": sum(r.get("clipped", False) for r in group) / len(group) if group else None}
    shell = [r for r in records if r["kind"] == "shell"]
    body = sum(r["body_tokens"] for r in shell)
    result["truncation"] = {
        "body_truncated_rate": sum(r["body_truncated"] for r in shell) / len(shell) if shell else None,
        "end_token_coverage": sum(r["has_end_token"] for r in shell) / len(shell) if shell else None,
        "retained_body_ratio": sum(r["kept_body_tokens"] for r in shell) / body if body else None,
    }
    for label, predicate in (
        ("complete", lambda r: r["has_end_token"]),
        ("body_complete_without_end", lambda r: not r["body_truncated"] and not r["has_end_token"]),
        ("body_truncated", lambda r: r["body_truncated"]),
    ):
        group = [r for r in shell if predicate(r)]
        n = sum(r["tokens"] for r in group)
        result[label] = {"events": len(group), "nll": sum(r["nll_sum"] for r in group) / n if n else None}
        generated = [r for r in group if "body_match" in r]
        with_end = [r for r in generated if r["end_correct"] is not None]
        result[label].update(
            generated_events=len(generated),
            body_match_rate=sum(r["body_match"] for r in generated) / len(generated) if generated else None,
            exact_with_end_rate=sum(r["end_correct"] for r in with_end) / len(with_end) if with_end else None,
        )
    switches = [records[i + 2] for i in range(len(records) - 2)
                if records[i].get("command") == "close" and records[i + 1].get("command") == "open"
                and records[i + 2]["kind"] == "shell"]
    result["cross_boundary_triples"] = len(switches)
    result["post_switch_nll"] = (sum(r["nll_sum"] for r in switches) / sum(r["tokens"] for r in switches)) if switches else None
    return result
