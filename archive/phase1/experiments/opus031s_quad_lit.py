"""opus-0.3.1s — v0.2.20s base + quadratic residual corrector (120-dim).

Aggressive structural change to the corrector: the 14 hand features
(v0.2.14s–v0.2.20s) gain a per-unit sigma-scale feature (s/mean(s)) and
ALL pairwise products -> 120-dim ridge basis. θ refit on the 500-MLP
synth cache (N_GT=1.5e5) with 5-fold OOS model selection:
synth OOS ratio 1.027 (linear) -> 1.073 (quad), a ~4.5% honest held-out
MSE reduction expected to transfer to hidden. scale=1.0 (no
extrapolation) for transfer safety.

Base RQMC+CV+gain blend, λ=0.85, floor=0.094, reserve=2.0e9 unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_FLOOR_FRAC = 0.094
_MATMUL_PER_LAYER = 2.0
_PPF_EPS = 1e-6
_BLEND_RQMC = 0.85
_GAIN_SCALE = 0.992
_GAIN_RESERVE = 2.0e9
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_RIDGE_REL = 1.0e-4

# opus-0.3.0s corrector (experiments/corrector_opus030.npz): 120-dim quad
# basis, ridge θ, 5-fold OOS-selected (lam=1e-3, scale=1.0). Arrays inlined
# as base64(float64) so the estimator is self-contained (grader runs only
# estimator.py + imported modules; no data file ships).
_CORR_SCALE = 1.0
_CORR_BETA = fnp.asarray(
    [
        -1.1815019388450310e-04, -2.1297282073646784e-03, -1.7425357364118099e-03, -1.7256685532629490e-03,
        5.1144128519808874e-05, -5.8884627651423216e-04, 1.2032273843942676e-05, 1.7418379138689488e-04,
        6.7005242453888059e-04, -5.6159787345677614e-04, 3.6432669730857015e-04, -3.6432765773497522e-04,
        2.7952191885560751e-03, 2.8711539926007390e-04, -2.1839341206941754e-04, -1.7425357364118099e-03,
        -1.7256685532629490e-03, 5.1144128519808874e-05, 8.8947701442521065e-05, 1.3020453043282032e-03,
        -7.2762730997055769e-04, 1.4531992201227695e-04, -1.4703939086757600e-04, 4.5368561404757202e-04,
        -3.4738139947876334e-04, 3.4738166141323745e-04, -2.9517805160139687e-05, 7.8919698717072606e-04,
        -2.0922629628330469e-03, 5.1144128519808874e-05, 8.8947701442521065e-05, 6.1943585751578212e-04,
        1.5905892360024154e-04, 1.5554935089312494e-04, -5.9308437630534172e-04, -5.6458542530890554e-05,
        5.3971953457221389e-04, 5.4307159734889865e-04, -5.4307165555655956e-04, 1.4227075735107064e-03,
        -1.0785909398691729e-04, -4.4856747263111174e-04, 6.1943585751578212e-04, -3.7427482311613858e-04,
        5.6070525897666812e-04, -1.5115394489839673e-04, -2.9047939460724592e-04, 1.4545896556228399e-04,
        -2.7264413802186027e-05, -2.1926836780039594e-05, 2.1926836780039594e-05, 5.3410651162266731e-04,
        -5.5582128698006272e-04, 2.9346410883590579e-04, -5.8860675198957324e-04, 3.4492163104005158e-04,
        -7.5652724262909032e-06, 1.2490930384956300e-04, 2.8575721080414951e-04, -2.5504350196570158e-04,
        -8.1537582445889711e-05, 8.1537582445889711e-05, 2.1526659838855267e-03, -3.2743968768045306e-04,
        1.5523280308116227e-04, -1.0358473518863320e-03, 3.6740713403560221e-04, -5.2290968596935272e-04,
        -1.4090295007918030e-04, 4.3809963244711980e-05, 1.7340545309707522e-04, -1.7340837803203613e-04,
        -7.6300004730001092e-04, -6.2737934058532119e-05, 2.4669598133186810e-05, 1.7221315647475421e-04,
        4.7861365601420403e-04, 1.5239939966704696e-04, 2.5838345754891634e-04, 1.3794827100355178e-04,
        -1.3795573613606393e-04, -1.8380864639766514e-04, 7.4032745033036917e-05, -1.0836783330887556e-03,
        5.2722709369845688e-05, -5.1885776883864310e-06, -2.4974297048174776e-05, -1.9302708096802235e-04,
        1.9309367053210735e-04, -8.2146970089524984e-04, 5.7617429411038756e-04, -7.1370770456269383e-04,
        9.2184651293791831e-05, 6.2945895479060709e-05, -6.0518959799082950e-05, 6.0540929553098977e-05,
        5.9390801470726728e-04, -1.3431586557999253e-04, -2.0449495059438050e-04, -1.0100833378601237e-06,
        -1.6701656568329781e-04, 1.6717344988137484e-04, -3.2142977579496801e-04, 4.5498210238292813e-04,
        -1.7824861060944386e-05, -1.7703762568999082e-04, 1.7710606334730983e-04, 3.4563770168460906e-04,
        -3.3681018976494670e-04, 2.9680880834348500e-04, -1.7711814143694937e-04, -3.4563802182674408e-04,
        3.3681583590805531e-04, -2.9680965235456824e-04, 2.5775837711989880e-03, -1.3729702914133668e-03,
        1.4866882702335715e-03, 9.7157451091334224e-04, 3.7800587597303092e-04, 2.4181952176149935e-04,
    ],
    dtype=fnp.float32,
)
_CORR_MU = fnp.asarray(
    [
        0.0000000000000000e+00, -1.6919450135901570e-03, 1.7505096435546875e+01, -1.9709719717502594e-01,
        5.3856927490234375e+02, 6.2968611717224121e-02, -4.9266743008047342e-04, -2.0958119421266019e-04,
        -7.7920221883687191e-06, 1.5935821284074336e-04, -8.2195423601660877e-05, 4.6577406465075910e-04,
        -1.3558421051129699e-03, 6.5239347517490387e-02, 1.0000000000000000e+00, 1.7505096435546875e+01,
        -1.9709719717502594e-01, 5.3856927490234375e+02, -1.2532669067382812e+01, -2.6378702023066580e-04,
        1.2020542053505778e-03, -3.5046097764279693e-05, -2.0845460239797831e-03, -1.4600007034459850e-06,
        1.8723783432506025e-04, -1.0610143654048443e-03, 1.4458148956298828e+01, -2.3946199507918209e-04,
        -3.0523720197379589e-03, 5.3856927490234375e+02, -1.2532669067382812e+01, 2.0260755859375000e+04,
        6.5169326961040497e-02, 5.3111365996301174e-03, 1.4453585026785731e-03, -3.5852219752996461e-06,
        3.3431923948228359e-03, 8.1384502118453383e-04, -4.6117883175611496e-03, -1.4892581105232239e-01,
        7.7365435659885406e-02, 2.1662872314453125e+01, 2.0260755859375000e+04, -7.0456903076171875e+02,
        -1.6387223149649799e-04, 3.9603941142559052e-02, -1.0998838115483522e-03, -7.6862527057528496e-03,
        1.9107706975773908e-05, 5.9841391630470753e-03, -3.3910121768712997e-02, 4.2858221435546875e+02,
        -9.9574521300382912e-05, -3.0085769295692444e-01, 8.5962250000000000e+05, 2.0095215737819672e-01,
        2.3166280984878540e-01, 1.0621558874845505e-01, -3.4601354855112731e-05, 2.8181901201605797e-02,
        3.4738987684249878e-02, -1.9685426354408264e-01, -9.4141864776611328e+00, 3.0146521329879761e-01,
        7.3462322998046875e+02, 1.7593009397387505e-02, -1.3921510253567249e-04, -3.8623431464657187e-05,
        -3.1676952403358882e-06, -1.3057459727860987e-04, -2.4389753889408894e-05, 1.3820860476698726e-04,
        -2.7005877927877009e-04, 1.7757572233676910e-02, 4.5751389116048813e-02, 4.9786450108513236e-04,
        3.3052794606192037e-05, 4.5621900426340289e-06, -3.7251805196092391e-08, 6.8068940890952945e-05,
        -3.8572397897951305e-04, 1.6934007871896029e-03, -1.5822863497305661e-04, -1.6287479957100004e-04,
        4.6899942390155047e-05, 1.3031163348387054e-07, -1.4787721624998085e-07, 4.3367813304939773e-06,
        -2.4575094357714988e-05, -3.1188083084998652e-05, -4.0999868360813707e-05, -2.1837926760781556e-04,
        3.3765154512366280e-05, 1.5196266645034484e-07, 6.7679218318517087e-07, -3.8351558941940311e-06,
        -1.9781184382736683e-03, -3.2684265534044243e-06, -5.1755077947746031e-06, 5.3448816288437229e-06,
        3.9916780991688938e-08, -2.2619508399657207e-07, -2.1260161702230107e-06, -1.2621073983609676e-04,
        1.5869551862124354e-04, 9.3912449301569723e-06, -5.3217056120047346e-05, 2.4848841712810099e-04,
        -2.7136531571159139e-05, -2.9570570404757746e-05, 3.0156329739838839e-04, -1.4081010594964027e-03,
        1.5377368254121393e-04, 1.6756655531935394e-04, 1.2038200378417969e+01, -2.4656820460222661e-04,
        -2.3848954588174820e-03, 1.7972771078348160e-02, 4.7648720443248749e-02, 1.1461553573608398e+00,
    ],
    dtype=fnp.float32,
)
_CORR_SD = fnp.asarray(
    [
        1.0000000000000000e+00, 4.1839089393615723e+00, 1.5236169815063477e+01, 1.4234014892578125e+02,
        7.5469573974609375e+02, 1.1673887819051743e-01, 2.2307448089122772e-02, 6.8451557308435440e-03,
        5.8107841759920120e-03, 2.3064112756401300e-03, 3.0634212307631969e-03, 1.7359340563416481e-02,
        3.4696106910705566e+00, 1.1711788922548294e-01, 3.8230267167091370e-01, 1.5236169815063477e+01,
        1.4234014892578125e+02, 7.5469573974609375e+02, 6.3049516601562500e+03, 9.4793938100337982e-02,
        8.5414119064807892e-02, 2.5158021599054337e-02, 4.7221016138792038e-03, 3.0147654470056295e-03,
        1.0787097737193108e-02, 6.1126839369535446e-02, 1.1588902473449707e+01, 9.9562905728816986e-02,
        5.4826498031616211e+00, 7.5469573974609375e+02, 6.3049516601562500e+03, 3.9444148437500000e+04,
        9.7337000072002411e-02, 4.5475113391876221e-01, 1.3019861280918121e-01, 7.7889021486043930e-03,
        6.1530028469860554e-03, 5.5017031729221344e-02, 3.1176313757896423e-01, 1.1083570861816406e+02,
        1.0539464652538300e-01, 2.5264257431030273e+01, 3.9444148437500000e+04, 3.2061256250000000e+05,
        1.8644477427005768e-01, 2.7225348949432373e+00, 7.6282888650894165e-01, 1.3064811937510967e-02,
        1.8587710335850716e-02, 3.2020261883735657e-01, 1.8144814968109131e+00, 5.6391424560546875e+02,
        2.2674460709095001e-01, 2.1602870178222656e+02, 2.2118662500000000e+06, 2.8758847713470459e-01,
        1.7449634552001953e+01, 4.8047299385070801e+00, 3.5213395953178406e-02, 4.7206398099660873e-02,
        2.0092163085937500e+00, 1.1385559082031250e+01, 4.7417832031250000e+03, 3.7126171588897705e-01,
        1.2679747314453125e+03, 4.0737532079219818e-02, 3.2505409326404333e-03, 1.0507848346605897e-03,
        1.8209173576906323e-03, 6.9289706880226731e-04, 4.8665772192180157e-04, 2.7576803695410490e-03,
        9.3274027109146118e-02, 4.0810696780681610e-02, 8.6714558303356171e-02, 1.1039375094696879e-03,
        1.7294712597504258e-04, 1.5500704466830939e-04, 6.0038531955797225e-05, 1.5199097106233239e-04,
        8.6123554501682520e-04, 7.1338161826133728e-02, 3.2975862268358469e-03, 2.1983258426189423e-02,
        8.3105354860890657e-05, 4.6234945330070332e-05, 1.9639281163108535e-05, 2.3881033484940417e-05,
        1.3527918781619519e-04, 2.1259956061840057e-02, 1.0606610449030995e-03, 5.9516103938221931e-03,
        1.3577482604887336e-04, 4.7662098950240761e-05, 2.2692300262860954e-05, 1.2854303349740803e-04,
        4.4883508235216141e-03, 1.8436347600072622e-03, 4.3850964866578579e-03, 2.5980969439842738e-05,
        8.7754269770812243e-06, 4.9680751544656232e-05, 2.8022343758493662e-03, 6.9793290458619595e-04,
        1.6461008926853538e-03, 2.1312642275006510e-05, 1.2072497338522226e-04, 9.1653754934668541e-03,
        4.9258500803261995e-04, 2.8814040124416351e-03, 6.8406149512156844e-04, 5.1937080919742584e-02,
        2.7912682853639126e-03, 1.6327908262610435e-02, 8.9391393661499023e+00, 9.7064927220344543e-02,
        4.3987913131713867e+00, 4.0929976850748062e-02, 8.7249331176280975e-02, 9.9184793233871460e-01,
    ],
    dtype=fnp.float32,
)


def _roberts_alpha(dim: int) -> fnp.ndarray:
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            lhs = math.exp(dim * math.log(mid))
        except OverflowError:
            lhs = float("inf")
        if lhs - mid - 1.0 > 0.0:
            hi = mid
        else:
            lo = mid
    log_phi = math.log(lo)
    alphas = [math.exp(-(j + 1) * log_phi) for j in range(dim)]
    return fnp.asarray(alphas, dtype=fnp.float32)


def _rqmc_standard_normal(n_samples: int, dim: int, seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed)
    shift = rng.random(dim).astype(fnp.float32)
    alpha = _roberts_alpha(dim)
    idx = fnp.arange(1, n_samples + 1, dtype=fnp.float32).reshape(n_samples, 1)
    raw = idx * alpha.reshape(1, dim) + shift.reshape(1, dim)
    u = raw - fnp.floor(raw)
    u = fnp.clip(u, _PPF_EPS, 1.0 - _PPF_EPS)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def _multi_cv_mean(
    h: fnp.ndarray, controls: fnp.ndarray, expectations: fnp.ndarray, ridge_rel: float
) -> fnp.ndarray:
    n = float(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc = controls - expectations.reshape(1, -1)
    gram = (cc.T @ cc) / n
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((h - mu).T @ cc) / n
    beta = fnp.linalg.solve(gram.T, cov.T).T
    return mu - beta @ fnp.mean(cc, axis=0)


def _gain_and_last_stats(weights: list):
    """Gain-cov final mean + last-layer (α, φ, σ)."""
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    alpha = phi = sigma = None
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)
        gain = fnp.where(sigma > 1e-12, Phi, fnp.zeros_like(Phi))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
    return mu, alpha, phi, sigma


class Estimator(BaseEstimator):
    """Roberts RQMC + L1+sq CV + gain blend + synth residual corrector."""

    VERSION = "opus-0.3.1s"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)

        blend = _BLEND_RQMC
        reserve = _GAIN_RESERVE if blend < 1.0 else 0.0
        flops_per_sample = depth * _MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(budget) * _FLOOR_FRAC - reserve, flops_per_sample)
        n_samples = int(max(64, min(target / flops_per_sample, 200_000)))

        def _stream_parts(n_half: int, seed: int):
            x = _rqmc_standard_normal(n_half, width, seed=seed)
            x = fnp.concatenate([x, -x], axis=0)
            w0 = weights[0]
            a1 = fnp.maximum(x @ w0, 0.0)
            h = a1
            for w in weights[1:-1]:
                h = fnp.maximum(h @ w, 0.0)
            zpre_s = h @ weights[-1]
            last = fnp.maximum(zpre_s, 0.0)
            return a1, last, zpre_s

        seed0 = int(mlp.seed) & 0x7FFFFFFF
        n_half = max(32, n_samples // 4)
        a1_a, last_a, z1 = _stream_parts(n_half, seed0)
        a1_b, last_b, z2 = _stream_parts(n_half, seed0 ^ 0xA5A5A5A5)
        a1 = fnp.concatenate([a1_a, a1_b], axis=0)
        last = fnp.concatenate([last_a, last_b], axis=0)
        zpre = fnp.concatenate([z1, z2], axis=0)
        w0 = weights[0]
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * _INV_SQRT_2PI
        q1 = fnp.sum(a1 * a1, axis=1)
        e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
        expectations = fnp.concatenate(
            [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
        )
        mu_cv = _multi_cv_mean(last, controls, expectations, _RIDGE_REL)

        mu_g_raw, alpha_g, phi_g, sig_g = _gain_and_last_stats(weights)
        mu_g = _GAIN_SCALE * mu_g_raw
        pred = blend * mu_cv + (1.0 - blend) * mu_g

        # Sample last-preact moments for features / σ scale.
        zc = zpre - fnp.mean(zpre, axis=0)
        m2 = fnp.maximum(fnp.mean(zc * zc, axis=0), 1e-12)
        m3 = fnp.mean(zc * zc * zc, axis=0)
        m4 = fnp.mean(zc * zc * zc * zc, axis=0)
        s = fnp.sqrt(m2)
        skew = m3 / (m2 * s)
        kurt = m4 / (m2 * m2) - 3.0
        alpha_s = fnp.mean(zpre, axis=0) / s
        phi_s = fnp.exp(-0.5 * alpha_s * alpha_s) * _INV_SQRT_2PI
        st = -(skew / 6.0) * phi_s
        kt = (kurt / 24.0) * (alpha_s * alpha_s - 1.0) * phi_s
        gap = mu_cv - mu_g
        glayer = fnp.full(width, float(fnp.mean(gap)), dtype=fnp.float32)

        ones = fnp.ones(width, dtype=fnp.float32)
        a = alpha_g
        sig_scale = s / (fnp.mean(s) + 1e-12)
        # 15 base features (14 legacy + sigma-scale); order MUST match the
        # trainer (experiments/train_corrector_opus030.py::build_feats).
        base_rows = [
            ones,
            a,
            a * a,
            a * a * a,
            a * a * a * a,
            phi_g,
            gap / (sig_g + 1e-8),
            glayer / (sig_g + 1e-8),
            st,
            kt,
            (mu_cv - pred) / (s + 1e-8),
            (mu_g - pred) / (s + 1e-8),
            alpha_s,
            phi_s,
            sig_scale,
        ]
        k = len(base_rows)  # 15
        rows = list(base_rows)  # linear part
        for i in range(1, k):
            for j in range(i, k):
                rows.append(base_rows[i] * base_rows[j])  # pairwise products
        feats = fnp.stack(rows, axis=0)  # (120, width)
        # standardize per feature, apply θ
        Xs = (feats - _CORR_MU.reshape(-1, 1)) / _CORR_SD.reshape(-1, 1)
        delta = _CORR_SCALE * s * (_CORR_BETA @ Xs)
        # Harden against rare explosions on tiny smoke MLPs.
        delta = fnp.clip(delta, -10.0 * s, 10.0 * s)
        mu_final = fnp.maximum(pred + delta, 0.0)

        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(f"Could not find baseline `{name}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print(f"--- RQMC+CV+blend+corr (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))
    if args.baseline:
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(
            _load_baseline(args.baseline)(), mlp, estimator_budget=int(2.72e11)
        )
