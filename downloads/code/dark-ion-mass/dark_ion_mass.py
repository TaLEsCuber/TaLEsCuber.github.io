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
DEFAULT_F0_HZ = 448250.0
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


def axial_displacements(mass_u, reference_u=REFERENCE_U):
    """Physical displacement vectors, Euclidean normalized, bright ion first.

    These are not the orthonormal eigenvectors in mass-weighted coordinates.
    Each vector's overall sign is arbitrary; the bright component is positive.
    """
    frequencies = axial_modes(mass_u, reference_u, 1.0)
    ratios = [2 - f * f for f in frequencies]
    return tuple((1 / math.hypot(1, r), r / math.hypot(1, r)) for r in ratios)


def chain_modes(masses_u, equilibrium_scaled, reference_u=REFERENCE_U,
                f0_hz=DEFAULT_F0_HZ):
    """Axial Hessian at supplied equilibrium positions x=z/(C/k_z)^(1/3).

    Requires NumPy. Returns Hz and columns of mass-weighted orthonormal vectors.
    All charges are +e; all ions share the same axial DC spring constant.
    This function verifies force balance but does not solve for positions.
    """
    import numpy as np
    masses = np.asarray(masses_u, dtype=float)
    x = np.asarray(equilibrium_scaled, dtype=float)
    positive_finite(reference_u, f0_hz, *masses)
    if masses.ndim != 1 or x.shape != masses.shape or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError('Supply equal-length finite one-dimensional mass and position arrays')
    stiffness = np.eye(len(x))
    force_balance = x.copy()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            separation = x[i] - x[j]
            if separation == 0:
                raise ValueError('Equilibrium positions must be distinct')
            coupling = 2 / abs(separation)**3
            stiffness[i, i] += coupling
            stiffness[j, j] += coupling
            stiffness[i, j] -= coupling
            stiffness[j, i] -= coupling
            repulsion = separation / abs(separation)**3
            force_balance[i] -= repulsion
            force_balance[j] += repulsion
    if np.max(np.abs(force_balance)) > 1e-7:
        raise ValueError('Positions do not satisfy axial equilibrium in the chosen units')
    inverse_sqrt_mu = np.sqrt(reference_u / masses)
    dynamical = stiffness * np.outer(inverse_sqrt_mu, inverse_sqrt_mu)
    eigenvalues, vectors = np.linalg.eigh(dynamical)
    if np.any(eigenvalues <= 0):
        raise ValueError('Axial equilibrium must have positive squared mode frequencies')
    return f0_hz * np.sqrt(eigenvalues), vectors


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
        for f, vector in zip(axial_modes(mu * REFERENCE_U, REFERENCE_U, 1),
                             axial_displacements(mu * REFERENCE_U)):
            v = np.asarray(vector)
            assert np.allclose(np.array([[2., -1.], [-1., 2.]]) @ v,
                               f**2 * np.diag([1., mu]) @ v, atol=1e-12)
    s = (5 / 4)**(1 / 3)
    frequencies, vectors = chain_modes([REFERENCE_U] * 3, [-s, 0, s])
    assert np.allclose(frequencies / DEFAULT_F0_HZ, np.sqrt([1, 3, 29 / 5]))
    assert np.allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)
    masses = np.array([REFERENCE_U, CANDIDATES['172Yb16OH+'], REFERENCE_U])
    frequencies, vectors = chain_modes(masses, [-s, 0, s])
    stiffness = np.array([[14, -8, -1], [-8, 21, -8], [-1, -8, 14]]) / 5
    physical_vectors = vectors / np.sqrt(masses / REFERENCE_U)[:, None]
    assert np.allclose(stiffness @ physical_vectors,
                       (masses / REFERENCE_U)[:, None] * physical_vectors
                       * (frequencies / DEFAULT_F0_HZ)**2)
    assert math.isclose(frequencies[1], DEFAULT_F0_HZ * math.sqrt(3))
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
    print("Verified: two/three-ion Hessians, physical mode vectors, both inversions, equal-mass limits, error propagation, input bounds")


def figures(output, f0_hz=DEFAULT_F0_HZ, reference_u=REFERENCE_U):
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
    positive_finite(f0_hz, reference_u)
    f0_khz = f0_hz / 1000
    mass = np.linspace(reference_u, reference_u + 36, 600)
    mu = mass / reference_u
    modes = np.array([axial_modes(m, reference_u, 1) for m in mass])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout='constrained')
    axes[0].plot(mass, 1 / mu, label='单离子：RF 主导径向', color='#d97706')
    axes[0].plot(mass, 1 / np.sqrt(mu), label='单离子：轴向 DC', color='#0f766e')
    axes[0].plot(mass, modes[:, 0], label='双离子：轴向低频模', color='#2563eb')
    axes[0].set(xlabel='暗离子质量 / u', ylabel='相对于变暗前同类频率的比值',
                title='A  不同测量对象，频率缩放不同')
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    axes[1].plot(mass, modes[:, 0] * f0_khz, label='低频同相模', color='#2563eb')
    for label, m in CANDIDATES.items():
        lower, _ = axial_modes(m, reference_u, f0_khz)
        axes[1].plot(m, lower, 'o', color='#2563eb')
    axes[1].set(xlabel='暗离子质量 / u', ylabel='双离子低频模 / kHz',
                title=f'B  探针质量 {reference_u:.3f} u，f0 = {f0_khz:g} kHz')
    shift = f0_khz - axial_modes(reference_u + H_U, reference_u, f0_khz)[0]
    axes[1].text(0.05, 0.12, f'质量增加约 1 u\n频率降低约 {shift:.2f} kHz',
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
            f = axial_modes(m, reference_u, f0_khz)[0]
            xs.append(f)
            ax.vlines(f, y - 0.20, y + 0.20, linewidth=3, color=['#0f766e', '#b45309'][y])
            ax.text(f, y + 0.27, f'{f:.3f} kHz', ha='center', fontsize=11)
        delta = abs(xs[0] - xs[1]) * 1000
        padding = .18 * f0_khz / 200
        ax.set(xlim=(min(xs) - padding, max(xs) + padding), ylim=(-.5, 1.75),
               yticks=[0, 1], yticklabels=[label for label, _ in group],
               xlabel='双离子低频轴向共振 / kHz', title=f'{title}：间隔 {delta:.1f} Hz')
        ax.grid(axis='x', alpha=.2)
        ax.ticklabel_format(axis='x', useOffset=False)
    fig.suptitle(f'候选的预测共振位置 · f0 = {f0_khz:g} kHz · 竖线不表示峰宽', fontsize=14)
    fig.savefig(output / 'candidate-frequency-separation.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), layout='constrained')
    for row, name in enumerate(('172Yb+', '172Yb16OH+')):
        m = CANDIDATES[name]
        frequencies = axial_modes(m, reference_u, f0_khz)
        for col, (f, vector) in enumerate(zip(frequencies, axial_displacements(m, reference_u))):
            ax = axes[row, col]
            ratio = vector[1] / vector[0]
            for x, component, color, label in zip((-1, 1), vector,
                                                   ('#2563eb', '#64748b'), ('亮探针', name)):
                ax.plot(x, 0, 'o', color=color, markersize=12)
                ax.annotate('', xy=(x + component * .85, .22), xytext=(x, .22),
                            arrowprops={'arrowstyle': '->', 'lw': 2.5, 'color': color})
                ax.text(x, -.24, label, ha='center', fontsize=11)
            ax.set(xlim=(-1.5, 2), ylim=(-.5, .65),
                   title=f'{"同相低频模" if col == 0 else "反相高频模"}  {f:.3f} kHz\n暗/亮位移比 = {ratio:.4f}')
            ax.axis('off')
    fig.suptitle(f'双离子本征振型 · f0 = {f0_khz:g} kHz\n箭头为同一时刻的相对位移；每个模单独归一化，非实测振幅', fontsize=13)
    fig.savefig(output / 'axial-mode-shapes.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-u', type=float, default=REFERENCE_U)
    parser.add_argument('--f0-hz', type=float, default=DEFAULT_F0_HZ)
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
        figures(args.figures, args.f0_hz, args.reference_u)
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
