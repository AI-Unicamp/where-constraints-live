r"""RL baseline: can a learned split policy beat the designed one at a matched budget?

The continuous search of exp464 has exactly one discrete decision without a
cheap exact evaluation: which piece to split when the level increases (the
designed policy: argmax of blame). This experiment runs a matched-compute
tournament on that decision, single instance (no training distribution,
the setting discussed in the thesis conclusion):

  designed  argmax blame (exp464 as-is), restarted with fresh seeds
  random    uniform piece choice, restarted with fresh seeds
  learned   REINFORCE over per-piece blame features (online, this instance)

Every arm consumes the same number of episodes; an episode is a full
k=2..k_target search (proposal + GNC descent per level). Metric per episode:
mean over shapes of (uncovered + outside + overlap) at the final level.
Report best-of-budget and mean of the last quarter.

Run (arms sequential):
  python experiments\exp466_rl_split_policy.py --names feec,heart ^
      --tag exp462_feec_heart_hero_holes_k20_alpha --episodes 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from exp462_zhou_multitarget_baco_2d import RESULTS, load_fg
from exp463_zhou_continuous_relax_2d import optimize
from exp464_zhou_continuous_search_2d import Search


class PolicyNet(torch.nn.Module):
    def __init__(self, n_feat: int = 5, width: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feat, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (k, n_feat) -> (k,)
        return self.net(feats).squeeze(-1)


def piece_features(search: Search, model) -> np.ndarray:
    """Per-piece features the designed heuristic also sees: blame split into
    outside / overlap / gap, plus area fraction and piece count context."""
    idx = model.clean_index(model.hard_index())
    k = search.k
    out = np.zeros((k, 5), dtype=np.float32)
    total = float(search.mask0.sum())
    for t in range(1, search.T):
        mem = model.memberships(idx, t)
        n = mem.sum(axis=0)
        m = search.fg[t].astype(bool)
        claimed = n > 0
        owner = np.argmax(mem, axis=0)
        for i in range(k):
            out[i, 0] += float((mem[i] & ~m).sum())
            out[i, 1] += float((mem[i] & (n >= 2) & m).sum())
        uncov = m & ~claimed
        if uncov.any() and claimed.any():
            from scipy.ndimage import distance_transform_edt
            _, (iy, ix) = distance_transform_edt(~claimed, return_indices=True)
            near = owner[iy, ix][uncov]
            for i in range(k):
                out[i, 2] += float((near == i).sum())
    for i in range(k):
        out[i, 3] = float((search.labels0 == i).sum()) / max(total, 1.0)
    out[:, 4] = k / 12.0
    # normalize blame columns by shape area so scale is episode-invariant
    out[:, :3] /= max(total, 1.0) / 100.0
    return out


def run_episode(fg, args, arm: str, policy, opt, rng) -> tuple[float, dict]:
    torch.manual_seed(int(rng.integers(2**31)))
    np.random.seed(int(rng.integers(2**31)))
    search = Search(fg, args)
    model = search.build_relax()
    logprobs = []
    while True:
        optimize(model, args, args.iters_level, log_every=10**9)
        search.absorb(model)
        model = search.build_relax()
        if search.k >= args.k_target:
            break
        feats = piece_features(search, model)
        if arm == "designed":
            choice = int(np.argmax(feats[:, :3].sum(axis=1)))
        elif arm == "random":
            choice = int(rng.integers(search.k))
        else:
            logits = policy(torch.from_numpy(feats))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logprobs.append(dist.log_prob(a))
            choice = int(a.item())
        # split the CHOSEN piece: replicate Search.split but with our choice
        blame_backup = Search.blame
        Search.blame = lambda self, m, c=choice, k=search.k: np.eye(k)[c]
        try:
            search.split(model)
        finally:
            Search.blame = blame_backup
        model = search.build_relax()
    vec = model.vec_metrics(search.labels0, fg)
    T = search.T
    J = float(np.mean([vec[f"s{t}_uncovered_pct"] + vec[f"s{t}_outside_pct"]
                       + vec[f"s{t}_overlap_pct"] for t in range(T)]))
    detail = {f"s{t}": {w: vec[f"s{t}_{w}_pct"] for w in
                        ("uncovered", "outside", "overlap")} for t in range(T)}
    if arm == "learned" and logprobs:
        return J, {"detail": detail, "logprobs": logprobs}
    return J, {"detail": detail}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="exp462_feec_heart_hero_holes_k20_alpha")
    p.add_argument("--names", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--arms", default="designed,random,learned")
    p.add_argument("--k-target", type=int, default=8)
    p.add_argument("--iters-level", type=int, default=250)
    p.add_argument("--iters-final", type=int, default=0)
    p.add_argument("--angles", type=int, default=12)
    p.add_argument("--peaks-per-angle", type=int, default=2)
    p.add_argument("--exact-evals", type=int, default=8)
    p.add_argument("--lr-policy", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    # GNC settings (exp464 values; tau0 kept mild)
    p.add_argument("--tau0", type=float, default=1.5)
    p.add_argument("--tau1", type=float, default=0.15)
    p.add_argument("--eps", type=float, default=1.5)
    p.add_argument("--w-cover", type=float, default=1.0)
    p.add_argument("--w-perim", type=float, default=0.03)
    p.add_argument("--w-minarea", type=float, default=1.0)
    p.add_argument("--w-pose", type=float, default=0.0)
    p.add_argument("--spill-weight", type=float, default=1.0)
    p.add_argument("--over-weight", type=float, default=3.0)
    p.add_argument("--w-well0", type=float, default=0.05)
    p.add_argument("--w-well1", type=float, default=10.0)
    p.add_argument("--min-area-frac", type=float, default=0.3)
    p.add_argument("--init-logit", type=float, default=2.0)
    p.add_argument("--lr-z", type=float, default=0.08)
    p.add_argument("--lr-pose", type=float, default=0.01)
    args = p.parse_args()

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    src = RESULTS / args.tag
    fg = [load_fg(src / "masks" / f"{n}.png") for n in names]
    run_dir = RESULTS / "exp466_rl_split_policy"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "episodes.jsonl"

    results: dict[str, list[float]] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        rng = np.random.default_rng(args.seed)
        policy = PolicyNet()
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
        baseline = None
        Js = []
        for ep in range(args.episodes):
            t0 = time.perf_counter()
            J, info = run_episode(fg, args, arm, policy, opt, rng)
            Js.append(J)
            if arm == "learned" and "logprobs" in info:
                baseline = J if baseline is None else 0.9 * baseline + 0.1 * J
                # lower J is better -> advantage = baseline - J
                loss = -(baseline - J) * torch.stack(info["logprobs"]).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            row = {"arm": arm, "episode": ep, "J": J,
                   "wall_s": time.perf_counter() - t0, **info["detail"]}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{arm}] ep {ep}: J={J:.2f} ({row['wall_s']:.0f}s)", flush=True)
        results[arm] = Js

    q = max(1, args.episodes // 4)
    summary = {
        arm: {"best": min(Js), "mean_last_quarter": float(np.mean(Js[-q:])),
              "mean_all": float(np.mean(Js)), "episodes": len(Js)}
        for arm, Js in results.items()
    }
    (run_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "summary": summary}, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
