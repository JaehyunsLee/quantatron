"""Step 4b: fault-tolerant cost of an active space.

Active-space integrals -> OpenFermion double factorization (+ lambda) -> Qualtran
DoubleFactorizationBlockEncoding (Toffolis per walk step, logical qubits). The number of
walk steps for qubitized QPE at fixed Delta E is ceil(pi * lambda / (2 * Delta E)), the same
convention as OpenFermion's compute_cost, which we also run as a cross-check.
"""

import math

import numpy as np
from openfermion.resource_estimates.df.compute_cost_df import compute_cost as of_df_cost
from openfermion.resource_estimates.df.factorize_df import factorize
from qualtran.bloqs.chemistry.df.double_factorization import DoubleFactorizationBlockEncoding
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value


def df_lambda(h1: np.ndarray, eri: np.ndarray, df_factors: np.ndarray) -> float:
    """Lambda of the DF Hamiltonian; same formula as openfermion compute_lambda_df, minus the pyscf_mf plumbing."""
    T = h1 - 0.5 * np.einsum("illj->ij", eri) + np.einsum("llij->ij", eri)
    lambda_T = np.sum(np.abs(np.linalg.eigvalsh(T)))
    lambda_F = sum(
        0.25 * np.sum(np.abs(np.linalg.eigvalsh(df_factors[:, :, k]))) ** 2 for k in range(df_factors.shape[2])
    )
    return float(lambda_T + lambda_F)


def _qualtran_step(n_spin_orb: int, L: int, Lxi: int, chi: int, beta_rot: int) -> tuple[int, int]:
    bloq = DoubleFactorizationBlockEncoding(
        num_spin_orb=n_spin_orb,
        num_aux=L,
        num_eig=Lxi,
        num_bits_state_prep=chi,
        num_bits_rot_aa_outer=1,
        num_bits_rot_aa_inner=7,
        num_bits_rot=beta_rot,
    )
    gates = get_cost_value(bloq, QECGatesCost())
    # A CSWAP and a logical AND each cost one Toffoli; the handful of single-qubit rotations is ignored.
    toffoli = gates.toffoli + gates.and_bloq + gates.cswap
    return int(toffoli), int(get_cost_value(bloq, QubitCount()))


def _with_padding(fn, L: int, Lxi: int, max_pad: int = 64):
    """Both cost models break for tiny ranks (QROM block sizes < 1). Pad the rank until they apply.

    Padding only adds data to load, so the result is a mild upper bound; the pad factor is recorded.
    """
    pad = 1
    while pad <= max_pad:
        try:
            return fn(L * pad, Lxi * pad), pad
        except (AssertionError, SystemExit, ValueError, ZeroDivisionError):
            pad *= 2
    raise RuntimeError(f"cost model failed even with rank padding x{max_pad}")


def estimate(h1: np.ndarray, eri: np.ndarray, cfg: dict) -> dict:
    """Toffolis, logical qubits and lambda for an active space with chemist-notation integrals."""
    norb = h1.shape[0]
    n_spin_orb = 2 * norb
    _, df_factors, L, Lxi = factorize(eri, cfg["df_threshold"])
    L, Lxi = max(int(L), 1), max(int(Lxi), 1)
    lam = df_lambda(h1, eri, df_factors)
    n_steps = math.ceil(math.pi * lam / (2 * cfg["delta_e"]))

    (q_step, q_qubits), q_pad = _with_padding(
        lambda l, lx: _qualtran_step(n_spin_orb, l, lx, cfg["chi"], cfg["beta_rot"]), L, Lxi
    )
    try:
        (of_step, of_total, of_qubits), of_pad = _with_padding(
            lambda l, lx: of_df_cost(n_spin_orb, lam, cfg["delta_e"], l, lx, cfg["chi"], cfg["beta_rot"], cfg["stps"]),
            L,
            Lxi,
        )
    except RuntimeError:
        of_step = of_total = of_qubits = of_pad = None

    return {
        "norb": norb,
        "lambda": lam,
        "df_rank": L,
        "df_num_eig": Lxi,
        "qpe_steps": n_steps,
        "toffoli_per_step": q_step,
        "toffolis": q_step * n_steps,
        "logical_qubits": q_qubits,
        "rank_pad": q_pad,
        "of_toffoli_per_step": of_step,
        "of_toffolis": of_total,
        "of_logical_qubits": of_qubits,
        "of_rank_pad": of_pad,
    }
