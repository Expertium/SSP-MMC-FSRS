"""Parity: the fused Triton value-iteration kernel must match the eager solver.

For each (user, hyperparameter set) we run the value iteration BOTH ways on identical inputs
and compare (a) the cost-to-go at a fixed iteration count -- isolating per-step numerical
agreement -- and (b) the actual convergence verdict (frac-at-max < 1/20) from the early-
stopping path. f32 reduction-order differences between the eager ``amin`` and the kernel's
``tl.min`` produce tiny value diffs; the verdict must be identical.

Run with:    uv run --no-sync python tests/test_solver7_kernel_parity.py
             uv run --no-sync python tests/test_solver7_kernel_parity.py 50 5   # users sets
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssp_mmc_fsrs.solver7 import SSPMMCSolver7, HAS_TRITON  # noqa: E402
from converge7 import (  # noqa: E402
    make_hyperparam_sets,
    load_jsonl_by_user,
    DEFAULT_PARAMS,
    DEFAULT_BUTTON_USAGE,
)

torch.set_num_threads(1)

FIXED_ITERS = (
    400  # run both engines this many steps (no early stop) to compare iterates
)


def main():
    if not HAS_TRITON:
        print("Triton not available -> nothing to compare. SKIP.")
        return 0

    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    common = sorted(set(params) & set(usage))
    common = [u for u in common if len(params[u]["parameters"]["0"]) == 34]
    user_ids = common[:n_users]
    hp_sets = make_hyperparam_sets(n_sets, seed=0)

    print(f"Comparing eager vs Triton on {len(user_ids)} users x {n_sets} sets")
    worst_abs = 0.0
    worst_rel = (
        0.0  # relative error, but only on "large" states (|cost| > 1) -- near-zero
    )
    # costs (e.g. the terminal at exactly 0) make relative error meaningless.
    verdict_mismatches = 0
    n_pairs = 0

    for user_id in user_ids:
        w = params[user_id]["parameters"]["0"]
        u = usage[user_id]
        solver = SSPMMCSolver7(
            review_costs=u["review_costs"],
            first_rating_prob=u["first_rating_prob"],
            review_rating_prob=u["review_rating_prob"],
            w=w,
        )
        for hp in hp_sets:
            n_pairs += 1
            with torch.inference_mode():
                cc = solver._const_cost(hp)
                # (a) fixed-iteration iterate comparison (tol=-1 -> never early-stop).
                se, _, _ = solver._run_iteration_eager(cc, FIXED_ITERS, -1.0)
                st, _, _ = solver._run_iteration_triton(cc, FIXED_ITERS, -1.0)
                abs_d = float((se - st).abs().max().item())
                large = se.abs() > 1.0
                rel_d = (
                    float(((se - st).abs()[large] / se.abs()[large]).max().item())
                    if bool(large.any())
                    else 0.0
                )
            worst_abs = max(worst_abs, abs_d)
            worst_rel = max(worst_rel, rel_d)
            # (b) verdict from the real early-stopping path.
            solver.engine = "eager"
            ve = solver.measure_convergence(hp)[0]
            solver.engine = "triton"
            vt = solver.measure_convergence(hp)[0]
            solver.engine = "auto"
            if ve != vt:
                verdict_mismatches += 1
                print(f"  VERDICT MISMATCH user {user_id}: eager={ve} triton={vt}")
        del solver
        torch.cuda.empty_cache()

    print(f"\npairs compared: {n_pairs}")
    print(
        f"worst |eager-triton| at iter {FIXED_ITERS}: abs={worst_abs:.2e}  "
        f"rel(|cost|>1)={worst_rel:.2e}"
    )
    print(f"frac-at-max verdict mismatches: {verdict_mismatches}")
    # The deliverable is the convergence verdict (must match exactly); the value diffs are
    # f32 reduction-order noise and must be tiny on non-trivial states.
    ok = verdict_mismatches == 0 and worst_abs < 1.0 and worst_rel < 1e-3
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
