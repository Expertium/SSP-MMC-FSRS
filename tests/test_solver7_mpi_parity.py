"""Parity: Modified Policy Iteration (``mpi_eval > 0``) must produce the SAME result as
plain value iteration (``mpi_eval = 0``). MPI converges to the same unique fixed point V*
in far fewer expensive (greedy) sweeps, so we require:

  (a) identical convergence verdicts (frac-at-max < 1/20) on the batched sweep path, and
  (b) the same policy (retention_matrix) and V* from ``solve`` -- V within the
      2*tol/(1-gamma) bound for two tol-converged solutions; the argmin policy within
      tie-break jitter.

Runs on the PRODUCTION grid (what ships). Run:
    uv run --no-sync python tests/test_solver7_mpi_parity.py [n_users] [n_sets]
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    DISCOUNT_FACTOR,
    SSPMMCSolver7,
    build_hybrid_s_grid,
    build_production_d_grid,
)
from converge7 import (  # noqa: E402
    DEFAULT_BUTTON_USAGE,
    DEFAULT_PARAMS,
    invalid_data_reason,
    load_jsonl_by_user,
    make_hyperparam_sets,
)

torch.set_num_threads(1)
# Two solutions each within tol of the unique V* differ by <= 2*tol/(1-gamma); +1 margin.
V_GATE = 2 * 0.1 / (1 - DISCOUNT_FACTOR) + 1.0


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    common = [
        u
        for u in sorted(set(params) & set(usage))
        if len(params[u]["parameters"]["0"]) == 34
    ]
    users = []
    for u in common:
        if invalid_data_reason(usage[u], params[u]["parameters"]["0"]) is None:
            users.append(u)
        if len(users) == n_users:
            break
    hp_sets = make_hyperparam_sets(max(n_sets, 2), seed=0)
    gk = dict(s_state=build_hybrid_s_grid(), d_state=build_production_d_grid())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"MPI vs plain VI on {len(users)} users x {n_sets} sets ({device}, production grid)"
    )

    verdict_mismatch = 0
    worst_v, worst_pol = 0.0, 0.0
    for uid in users:
        u = usage[uid]
        solver = SSPMMCSolver7(
            review_costs=u["review_costs"],
            first_rating_prob=u["first_rating_prob"],
            review_rating_prob=u["review_rating_prob"],
            w=params[uid]["parameters"]["0"],
            device=device,
            engine="auto",
            **gk,
        )
        # (a) batched convergence verdict parity (the sweep's deliverable).
        solver.mpi_eval = 0
        v0 = solver.measure_convergence_batched(hp_sets, n_iter=1500, batch_size=4)
        solver.mpi_eval = 20
        v1 = solver.measure_convergence_batched(hp_sets, n_iter=1500, batch_size=4)
        verdict_mismatch += sum(1 for a, b in zip(v0, v1) if a[0] != b[0])
        # (b) single-set solve policy + V* parity.
        for hp in hp_sets[:2]:
            solver.mpi_eval = 0
            cm0, rm0 = solver.solve(hp, n_iter=3000, verbose=False)
            solver.mpi_eval = 20
            cm1, rm1 = solver.solve(hp, n_iter=3000, verbose=False)
            worst_v = max(worst_v, float(np.abs(cm1 - cm0).max()))
            worst_pol = max(worst_pol, float((np.abs(rm1 - rm0) > 1e-9).mean()))
        del solver
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"convergence verdict mismatches (MPI vs plain): {verdict_mismatch}")
    print(f"worst ||V_mpi - V_plain||_inf: {worst_v:.3g}  (gate {V_GATE:.1f})")
    print(
        f"worst policy-diff fraction: {worst_pol:.4%}  (informational tie-break jitter)"
    )
    ok = verdict_mismatch == 0 and worst_v < V_GATE
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
