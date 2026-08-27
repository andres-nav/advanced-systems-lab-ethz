import argparse, struct
import numpy as np

def round_down(x, a):
    return (x // a) * a

def gen(seed, F, S, interval_regime, data_regime, align, block_size, rho, fill_invalid):
    rng = np.random.default_rng(seed)

    # interval length
    if interval_regime == "LONG":
        length = int(round(0.8 * S))
    elif interval_regime == "SHORT":
        length = int(round(0.3 * S))
    else:
        raise ValueError("interval_regime must be LONG or SHORT")

    if S >= align and align > 1:
        length = max(2, round_down(length, align))
    length = min(length, S)

    # starts/ends (aligned)
    starts = np.zeros(F, dtype=np.int32)
    ends   = np.zeros(F, dtype=np.int32)
    max_start = max(0, S - length)
    for i in range(F):
        s = 0 if max_start == 0 else int(rng.integers(0, max_start + 1))
        if align > 1:
            s = round_down(s, align)
        s = min(s, max_start)
        starts[i] = s
        ends[i]   = s + length

    X = np.full((F, S), fill_invalid, dtype=np.float32)

    if data_regime == "GAUSSIAN":
        for i in range(F):
            s, e = starts[i], ends[i]
            X[i, s:e] = rng.standard_normal(e - s).astype(np.float32)

    elif data_regime == "BLOCK_CORR":
        B = block_size
        num_groups = (F + B - 1) // B
        latent = rng.standard_normal((num_groups, S)).astype(np.float32)
        scale_noise = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        for i in range(F):
            g = i // B
            s, e = starts[i], ends[i]
            eps = rng.standard_normal(e - s).astype(np.float32)
            X[i, s:e] = rho * latent[g, s:e] + scale_noise * eps
    else:
        raise ValueError("data_regime must be GAUSSIAN or BLOCK_CORR")

    return X, starts, ends

def write_bin(path, X, starts, ends):
    F, S = X.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", F, S))
        f.write(starts.astype(np.int32).tobytes(order="C"))
        f.write(ends.astype(np.int32).tobytes(order="C"))
        f.write(X.astype(np.float32).tobytes(order="C"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--F", type=int, default=64)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--interval", choices=["LONG", "SHORT"], default="LONG")
    ap.add_argument("--data", choices=["GAUSSIAN", "BLOCK_CORR"], default="GAUSSIAN")
    ap.add_argument("--align", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--fill_invalid", type=float, default=0.0)
    args = ap.parse_args()

    X, starts, ends = gen(args.seed, args.F, args.S, args.interval, args.data,
                          args.align, args.block_size, args.rho, args.fill_invalid)

    write_bin(args.out, X, starts, ends)
    print(f"Wrote {args.out}: F={args.F}, S={args.S}")

if __name__ == "__main__":
    main()
