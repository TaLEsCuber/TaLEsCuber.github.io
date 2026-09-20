"""Two singly charged ions in a harmonic axial DC potential.

All CLI frequencies are Hz; masses are u. This is a mass estimator, not a
chemical identification engine. Statistical errors assume independent f0/fminus.
Neutral isotope masses: NIST Atomic Weights and Isotopic Compositions (Yb,H,O).
One electron is subtracted per singly charged ion; binding/ionization mass
corrections are negligible at the illustrative 0.1-u precision.
"""

import argparse
import math
from pathlib import Path

ELECTRON_U = 0.000548579909
REFERENCE_U = 171.9363859 - ELECTRON_U
H_U = 1.00782503223
O_U = 15.99491461957
CANDIDATES = {
    "172Yb+": REFERENCE_U,
    "172YbH+": REFERENCE_U + H_U,
    "172Yb16O+": REFERENCE_U + O_U,
    "172Yb16OH+": REFERENCE_U + O_U + H_U,
}


def positive_finite(*values):
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Masses and frequencies must be positive and finite")


def axial_modes(mass_u, reference_u, f0_hz):
    positive_finite(mass_u, reference_u, f0_hz)
    r = reference_u / mass_u
    upper_squared = 1 + r + math.sqrt(1 - r + r * r)
    lower_squared = 3 * r / upper_squared
    return f0_hz * math.sqrt(lower_squared), f0_hz * math.sqrt(upper_squared)


def infer_lower(fminus_hz, f0_hz, reference_u=REFERENCE_U):
    positive_finite(fminus_hz, f0_hz, reference_u)
    x = (fminus_hz / f0_hz) ** 2
    if not 0 < x < 1.5:
        raise ValueError("A positive-mass lower axial mode requires 0 < (f-/f0)^2 < 1.5")
    return reference_u * (3 - 2 * x) / (x * (2 - x))


def lower_mass_uncertainty(fminus_hz, f0_hz, sigma_minus_hz, sigma0_hz,
                           reference_u=REFERENCE_U):
    infer_lower(fminus_hz, f0_hz, reference_u)
    if any(not math.isfinite(v) or v < 0 for v in (sigma_minus_hz, sigma0_hz)):
        raise ValueError("Standard uncertainties must be finite and nonnegative")
    x = (fminus_hz / f0_hz) ** 2
    denominator = x * (2 - x)
    derivative = (-2 * denominator - (3 - 2 * x) * (2 - 2 * x)) / denominator**2
    sigma_x = 2 * x * math.hypot(sigma_minus_hz / fminus_hz, sigma0_hz / f0_hz)
    return abs(reference_u * derivative) * sigma_x


def infer_pair(fminus_hz, fplus_hz, f0_hz, reference_u=REFERENCE_U):
    positive_finite(fminus_hz, fplus_hz, f0_hz, reference_u)
    denominator = fplus_hz**2 + fminus_hz**2 - 2 * f0_hz**2
    if denominator <= 0 or fplus_hz <= fminus_hz:
        raise ValueError("Frequencies do not admit a positive two-ion mass estimate")
    # Necessary trace relation only: verify both eigenfrequencies afterwards.
    return 2 * reference_u * f0_hz**2 / denominator


def self_check():
    import numpy as np
    # Independent mass-weighted Hessian, including molecules lighter than the probe.
    for mu in (0.05, 0.5, 1.0, 1.006, 1.1, 2.0, 20.0):
        matrix = np.array([[2.0, -1 / math.sqrt(mu)],
                           [-1 / math.sqrt(mu), 2 / mu]])
        expected = 200000 * np.sqrt(np.linalg.eigvalsh(matrix))
        actual = axial_modes(mu * REFERENCE_U, REFERENCE_U, 200000)
        assert np.allclose(actual, expected, rtol=1e-12)
        assert math.isclose(infer_lower(actual[0], 200000), mu * REFERENCE_U, rel_tol=1e-12)
        assert math.isclose(infer_pair(*actual, 200000), mu * REFERENCE_U, rel_tol=1e-12)
    assert axial_modes(REFERENCE_U, REFERENCE_U, 200000)[0] == 200000
    rng = np.random.default_rng(172)
    fm, f0, sm, s0 = 195122.0, 200000.0, 25.0, 20.0
    draws = np.array([infer_lower(a, b) for a, b in zip(
        rng.normal(fm, sm, 40000), rng.normal(f0, s0, 40000))])
    predicted = lower_mass_uncertainty(fm, f0, sm, s0)
    assert abs(draws.std(ddof=1) / predicted - 1) < 0.02
    for invalid in (0, -1, float('nan'), float('inf'), 250000):
        try:
            infer_lower(invalid, 200000)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid frequency accepted")
    print("Verified: mass-weighted Hessian, both inversions, equal-mass limit, error propagation, input bounds")


def figures(output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        plt.rcParams['font.family'] = FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=True)
    mass = np.linspace(REFERENCE_U, REFERENCE_U + 36, 600)
    mu = mass / REFERENCE_U
    modes = np.array([axial_modes(m, REFERENCE_U, 1) for m in mass])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout='constrained')
    axes[0].plot(mass, 1 / mu, label='单离子：RF 主导径向', color='#d97706')
    axes[0].plot(mass, 1 / np.sqrt(mu), label='单离子：轴向 DC', color='#0f766e')
    axes[0].plot(mass, modes[:, 0], label='双离子：轴向低频模', color='#2563eb')
    axes[0].set(xlabel='暗离子质量 / u', ylabel='相对于变暗前同类频率的比值',
                title='A  不同测量对象，频率缩放不同')
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    axes[1].plot(mass, modes[:, 0] * 200, label='低频同相模', color='#2563eb')
    for label, m in CANDIDATES.items():
        lower, _ = axial_modes(m, REFERENCE_U, 200)
        axes[1].plot(m, lower, 'o', color='#2563eb')
    axes[1].set(xlabel='暗离子质量 / u', ylabel='双离子低频模 / kHz',
                title='B  亮探针为 172Yb+，单离子 f0 = 200 kHz')
    axes[1].text(0.05, 0.12, '质量增加约 1 u\n频率只降低约 0.29 kHz',
                 transform=axes[1].transAxes, color='#1d4ed8')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('理想谐势计算 · 单电荷 · 非实验数据', fontsize=14)
    fig.savefig(output / 'mass-frequency-map.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), layout='constrained')
    groups = [('原子离子与氢化物', list(CANDIDATES.items())[:2]),
              ('氧化物与氢氧化物', list(CANDIDATES.items())[2:])]
    for ax, (title, group) in zip(axes, groups):
        xs = []
        for y, (label, m) in enumerate(group):
            f = axial_modes(m, REFERENCE_U, 200)[0]
            xs.append(f)
            ax.vlines(f, y - 0.20, y + 0.20, linewidth=3, color=['#0f766e', '#b45309'][y])
            ax.text(f, y + 0.27, f'{f:.3f} kHz', ha='center', fontsize=11)
        delta = abs(xs[0] - xs[1]) * 1000
        ax.set(xlim=(min(xs) - .18, max(xs) + .18), ylim=(-.5, 1.75),
               yticks=[0, 1], yticklabels=[label for label, _ in group],
               xlabel='双离子低频轴向共振 / kHz', title=f'{title}：间隔 {delta:.1f} Hz')
        ax.grid(axis='x', alpha=.2)
        ax.ticklabel_format(axis='x', useOffset=False)
    fig.suptitle('候选的预测共振位置 · f0 = 200 kHz · 竖线不表示峰宽', fontsize=14)
    fig.savefig(output / 'candidate-frequency-separation.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-u', type=float, default=REFERENCE_U)
    parser.add_argument('--f0-hz', type=float, default=200000.0)
    parser.add_argument('--lower-hz', type=float)
    parser.add_argument('--upper-hz', type=float)
    parser.add_argument('--sigma-lower-hz', type=float, default=0)
    parser.add_argument('--sigma0-hz', type=float, default=0)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    if args.figures:
        figures(args.figures)
    if args.lower_hz is not None:
        mass = infer_lower(args.lower_hz, args.f0_hz, args.reference_u)
        sigma = lower_mass_uncertainty(args.lower_hz, args.f0_hz,
                                      args.sigma_lower_hz, args.sigma0_hz, args.reference_u)
        print(f'Mass from lower mode: {mass:.6f} u; statistical standard uncertainty: {sigma:.6f} u')
        predicted = axial_modes(mass, args.reference_u, args.f0_hz)
        print(f'Predicted upper mode: {predicted[1]:.3f} Hz')
        if args.upper_hz is not None:
            print(f'Observed minus predicted upper frequency: {args.upper_hz - predicted[1]:.3f} Hz')
            print(f'Mass from trace relation: {infer_pair(args.lower_hz, args.upper_hz, args.f0_hz, args.reference_u):.6f} u')
        print('Systematic shifts and chemical identity are not evaluated by this calculation.')
    else:
        print('species mass/u isolated_axial/kHz radial_RF_only/kHz lower/kHz upper/kHz')
        for name, m in CANDIDATES.items():
            lo, hi = axial_modes(m, args.reference_u, args.f0_hz)
            print(f'{name:12} {m:.6f} {args.f0_hz / math.sqrt(m / args.reference_u) / 1000:.3f} '
                  f'{1000 * args.reference_u / m:.3f} {lo / 1000:.3f} {hi / 1000:.3f}')
        print('Radial column assumes the reference RF-only radial frequency is 1000 kHz.')


if __name__ == '__main__':
    main()
