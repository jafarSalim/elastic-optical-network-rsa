"""Physical layer: closed-form GN model and the modulation reach table.

Every lightpath is assumed to see a fully loaded C-band (worst case), so the
generalised SNR depends only on the number of 80 km spans. Each span:

    P_ASE  = NF * h * nu * G * Rs                  (amplified spontaneous emission)
    P_NLI  = eta * P^3                             (Kerr nonlinear interference)
    eta    = (8/27) g^2 Leff^2 asinh(pi^2/2 |b2| La B^2) / (pi |b2| La) / Rs^2 * Rs

with La = 1/(2 alpha). Over N spans with incoherent accumulation
SNR_line = P / (N P_ASE + N eta P^3); at the optimum launch power
P* = (P_ASE / (2 eta))^(1/3) it equals P* / (1.5 N P_ASE). A transceiver
SNR ceiling is added in parallel: 1/GSNR = 1/SNR_line + 1/SNR_trx.

A format is allowed on a path if GSNR >= SNR_required(format) + margin,
where SNR_required gives a pre-FEC BER equal to the hard-decision FEC
threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

H = 6.62607015e-34
C = 299792458.0


@dataclass
class LinkParams:
    span_km: float = 80.0
    alpha_db_km: float = 0.20
    gamma: float = 1.3e-3          # 1/(W m)
    D_ps_nm_km: float = 17.0
    lam_nm: float = 1550.0
    nf_db: float = 5.0
    band_hz: float = 4.0e12        # occupied C-band
    rs_hz: float = 32e9            # reference symbol rate for SNR (cancels out)
    snr_trx_db: float = 25.0       # transceiver back-to-back SNR
    margin_db: float = 2.0         # system + ageing margin
    ber_fec: float = 3.8e-3        # hard-decision FEC threshold (7 % OH)


FORMATS = [  # name, bits per symbol, M
    ("BPSK", 1, 2), ("QPSK", 2, 4), ("8QAM", 3, 8),
    ("16QAM", 4, 16), ("32QAM", 5, 32), ("64QAM", 6, 64),
]


def qfunc(x):
    return 0.5 * erfc(np.asarray(x) / math.sqrt(2))


def ber_awgn(snr, M):
    """Approximate Gray-coded BER at linear SNR (Es/N0) for BPSK and M-QAM."""
    snr = np.asarray(snr, dtype=float)
    if M == 2:
        return qfunc(np.sqrt(2 * snr))
    if M == 4:
        return qfunc(np.sqrt(snr))
    k = math.log2(M)
    return (4 / k) * (1 - 1 / math.sqrt(M)) * qfunc(np.sqrt(3 * snr / (M - 1)))


def snr_required_db(M, ber):
    lo, hi = -10.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ber_awgn(10 ** (mid / 10), M) > ber:
            lo = mid
        else:
            hi = mid
    return hi


def span_noise(p: LinkParams):
    alpha = p.alpha_db_km / (10 * math.log10(math.e)) / 1e3        # 1/m
    L = p.span_km * 1e3
    leff = (1 - math.exp(-alpha * L)) / alpha
    la = 1 / (2 * alpha)
    beta2 = abs(p.D_ps_nm_km * 1e-6 * (p.lam_nm * 1e-9) ** 2 / (2 * math.pi * C))  # s^2/m
    G = 10 ** (p.alpha_db_km * p.span_km / 10)
    nu = C / (p.lam_nm * 1e-9)
    p_ase = 10 ** (p.nf_db / 10) * H * nu * G * p.rs_hz
    g_nli = (8 / 27) * p.gamma ** 2 * leff ** 2 * math.asinh(
        0.5 * math.pi ** 2 * beta2 * la * p.band_hz ** 2) / (math.pi * beta2 * la)
    eta = g_nli / p.rs_hz ** 2          # P_NLI = eta * P^3 for a channel of width Rs
    return p_ase, eta


def gsnr_db(n_spans, p: LinkParams = LinkParams(), launch_dbm=None):
    """GSNR after n_spans; optimum launch power unless launch_dbm is given."""
    p_ase, eta = span_noise(p)
    n = np.asarray(n_spans, dtype=float)
    if launch_dbm is None:
        P = (p_ase / (2 * eta)) ** (1 / 3)
    else:
        P = 10 ** ((np.asarray(launch_dbm) - 30) / 10)
    snr_line = P / (n * p_ase + n * eta * P ** 3)
    snr_trx = 10 ** (p.snr_trx_db / 10)
    return 10 * np.log10(1 / (1 / snr_line + 1 / snr_trx))


def optimum_launch_dbm(p: LinkParams = LinkParams()):
    p_ase, eta = span_noise(p)
    return 10 * math.log10((p_ase / (2 * eta)) ** (1 / 3)) + 30


def reach_table(p: LinkParams = LinkParams(), max_spans=200):
    """Maximum number of spans (and km) for each format."""
    n = np.arange(1, max_spans + 1)
    g = gsnr_db(n, p)
    rows = []
    for name, bits, M in FORMATS:
        need = snr_required_db(M, p.ber_fec) + p.margin_db
        ok = n[g >= need]
        spans = int(ok.max()) if ok.size else 0
        rows.append(dict(name=name, bits=bits, snr_req_db=snr_required_db(M, p.ber_fec),
                         need_db=need, max_spans=spans, reach_km=spans * p.span_km))
    return rows


class ModulationPicker:
    """Highest-order format whose reach covers a path (distance-adaptive)."""

    def __init__(self, p: LinkParams = LinkParams(), fixed=None):
        self.table = reach_table(p)
        self.fixed = fixed

    def pick(self, spans):
        if self.fixed is not None:
            row = next(r for r in self.table if r["name"] == self.fixed)
            return row if spans <= row["max_spans"] else None
        best = None
        for r in self.table:
            if spans <= r["max_spans"]:
                best = r
        return best


def slots_needed(rate_gbps, bits, slot_ghz=12.5, fec_rate=0.93, guard=1):
    """Contiguous slots for a demand: dual polarisation, one guard slot."""
    per_slot = slot_ghz * 2 * bits * fec_rate * 0.8   # 0.8: roll-off and pilots
    return int(math.ceil(rate_gbps / per_slot)) + guard
