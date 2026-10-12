"""Train the TopK SAE on dumped shards. Usage: train_sae.py LAYER [N_TOK].
Tracks MSE, L0, dead-feature rate over a sliding window. Saves models/sae.pt
+ results/sae_train.json. ~1-2h GPU for 2M tokens at 32k features.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sae.model import TopKSAE


def main() -> None:
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    dev = "cuda:0"
    m = TopKSAE(2048, 32768, 64).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, foreach=False)
    MICRO, ACCUM = 256, 4
    # init pre-bias to data mean (first shard sample)
    s0 = torch.load(min(Path(f"data/acts/l{layer}").glob("shard_*.pt")),
                    weights_only=True)["acts"]
    with torch.no_grad():
        m.pre_bias.data.copy_(s0[:50000].float().mean(0).to(dev))
    data_mean = s0[:50000].float().mean(0)
    del s0
    print("pre-bias init done", flush=True)
    act_counts = torch.zeros(32768)
    window = torch.zeros(32768)
    seen = win_mse = win_l0 = steps = resampled = 0
    t0 = time.time()
    for shard in sorted(Path(f"data/acts/l{layer}").glob("shard_*.pt")):
        print("  gpu GB before shard:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
        x = torch.load(shard, weights_only=True)["acts"].float()
        for b in range(0, len(x), 1024):
            opt.zero_grad()
            for q in range(b, min(b + 1024, len(x)), MICRO):
                xb = x[q:q + MICRO].float().to(dev)
                out = m(xb)
                (out["mse"] / ACCUM).backward()
                with torch.no_grad():
                    w = (out["acts"].detach().float().cpu() != 0).float().sum(0)
                    act_counts += w
                    window += w
                    seen += len(xb)
                    win_mse += float(out["mse"]) * len(xb)
                    win_l0 += float(out["l0"]) * len(xb)
                    steps += 1
                del out, xb
            opt.step()
            m.normalize_decoder()
        torch.cuda.empty_cache()
        with torch.no_grad():
            dead_now = window == 0
            resampled += m.resample_dead(dead_now.cuda(), data_mean)
            act_counts[dead_now] = 0  # judge reset features on new life only
            window.zero_()
        torch.cuda.empty_cache()
        print(f"  {shard.name} seen={seen} mse={win_mse / seen:.5f} "
              f"l0={win_l0 / seen:.1f} ({time.time() - t0:.0f}s)", flush=True)
        if seen >= n_tok:
            break
    dead = int((act_counts == 0).sum())
    # variance explained on last shard sample
    with torch.no_grad():
        xs = x[-20000:].float()
        var = float(((xs - xs.mean(0)) ** 2).mean())
        mses = []
        for q in range(0, len(xs), 2048):
            mses.append(float(m(xs[q:q + 2048].to(dev))["mse"]))
        fve = 1 - (sum(mses) / len(mses)) / var
    res = {"layer": layer, "tokens": seen, "mse": round(win_mse / seen, 6),
           "l0": round(win_l0 / seen, 2), "dead": dead, "d_sae": 32768,
           "dead_rate": round(dead / 32768, 4),
           "resampled": resampled,
           "var_explained": round(fve, 4),
           "secs": round(time.time() - t0)}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/sae_train_l{layer}.json").write_text(json.dumps(res, indent=2))
    Path("models").mkdir(exist_ok=True)
    torch.save(m.state(), f"models/sae_l{layer}.pt")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
