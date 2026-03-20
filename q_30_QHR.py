"""
QHR - Quantum Huber Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LAMBDA_REG = 0.01
HUBER_DELTA = 0.02
MAX_ITER = 200


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def huber_weights(residuals, delta=HUBER_DELTA):
    w = np.ones_like(residuals)
    mask = np.abs(residuals) > delta
    w[mask] = delta / (np.abs(residuals[mask]) + 1e-10)
    return w


def quantum_huber_regression(K, y, lam=LAMBDA_REG, delta=HUBER_DELTA,
                             max_iter=MAX_ITER):
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)

    for iteration in range(max_iter):
        pred = K @ alpha
        residuals = y - pred
        W = np.diag(huber_weights(residuals, delta))

        alpha_new = np.linalg.solve(
            K.T @ W @ K + lam * np.eye(n), K.T @ W @ y)

        if np.max(np.abs(alpha_new - alpha)) < 1e-8:
            alpha = alpha_new
            break
        alpha = alpha_new

    return K @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QHR po pozicijama (delta={HUBER_DELTA}, "
          f"lambda={LAMBDA_REG}, {MAX_ITER} iter) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = quantum_huber_regression(K, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QHR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QHR po pozicijama (delta=0.02, lambda=0.01, 200 iter) ---
  Poz 1 [1-33]: 1:0.167 | 2:0.146 | 3:0.129
  Poz 2 [2-34]: 8:0.086 | 5:0.076 | 9:0.076
  Poz 3 [3-35]: 13:0.064 | 12:0.063 | 14:0.062
  Poz 4 [4-36]: 23:0.064 | 21:0.063 | 18:0.063
  Poz 5 [5-37]: 29:0.065 | 26:0.064 | 27:0.063
  Poz 6 [6-38]: 33:0.084 | 32:0.081 | 35:0.080
  Poz 7 [7-39]: 7:0.182 | 38:0.152 | 37:0.132

==================================================
Predikcija (QHR, deterministicki, seed=39):
[1, 8, x, y, z, 33, 38]
==================================================
"""



"""
QHR - Quantum Huber Regression

QHR je kvantni algoritam za regresiju sa Huber regularizacijom.
QHR se sastoji od 5 qubita i 1 sloja Ry+CX+Rz rotacija.

Isti kvantni kernel (ZZFeatureMap, fidelity, 5 qubita)
Huber loss umesto MSE: za reziduale manje od delta=0.02 koristi kvadratni gubitak, za vece koristi linearni
IRLS (Iteratively Reweighted Least Squares): 200 iteracija sa adaptivnim tezinama iz Huber funkcije
Robusna regresija: otporna na outlier-e u empirijskoj distribuciji (retke/neobicne vrednosti ne dominiraju fitom)
Za razliku od Ridge koji tretira sve greske jednako, Huber smanjuje uticaj velikih devijacija
Deterministicki, brz, konvergira automatski
"""
