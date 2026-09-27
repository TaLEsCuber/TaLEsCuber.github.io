"""Reproducible 1D lamellar RCWA tutorial (TE and TM, exp(-i omega t)).

Run: python rcwa_demo.py
Dependencies: numpy, scipy, matplotlib. Lengths are in micrometres.
This is an educational scalar solver, not a 2D nanopillar/metalens solver.
Exterior media must be lossless, isotropic, nonmagnetic dielectrics.
Avoid exact modal cutoffs; production solvers need special cutoff handling.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eig, solve


def outgoing_sqrt(values):
    """Passive branch for exp(+i k0 q z); assumes ordinary dielectrics."""
    q = np.sqrt(np.asarray(values, dtype=complex))
    flip = (q.imag < -1e-10) | ((np.abs(q.imag) <= 1e-10) & (q.real < 0))
    return np.where(flip, -q, q)


def convolution(M, fill, eps_ridge, eps_gap):
    """Analytic coefficients of a centered ridge; differences span -2M..2M."""
    orders = np.arange(-M, M + 1)
    delta = orders[:, None] - orders[None, :]
    return (eps_ridge - eps_gap) * fill * np.sinc(delta * fill) + eps_gap * (delta == 0)


def layer_modes(K, E, B, pol):
    """State is (Ey,-Z0 Hx) for TE and (Z0 Hy,Ex) for TM."""
    I = np.eye(len(K))
    if pol == 'TE':
        values, W = eig(E - K @ K)
    elif pol == 'TM':
        # Li factorization: B = [[1/eps]], E = [[eps]], B != inv(E).
        values, W = eig(solve(B, I - K @ solve(E, K)))
    else:
        raise ValueError('pol must be TE or TM')
    q = outgoing_sqrt(values)
    if np.min(np.abs(q)) < 1e-9:
        raise ValueError('Exact modal cutoff requires a dedicated formulation.')
    V = W * q[None, :]
    if pol == 'TM':
        V = B @ V
    return W, V, q


def interface(Wl, Vl, Wr, Vr):
    """[b_left, a_right] = S [a_left, b_right]."""
    n = len(Wl)
    lhs = np.block([[Wl, -Wr], [Vl, Vr]])
    rhs = np.block([[-Wl, Wr], [Vl, Vr]])
    S = solve(lhs, rhs)
    return S[:n, :n], S[:n, n:], S[n:, :n], S[n:, n:]


def star(A, B):
    """Cascade A followed by B; use solves instead of explicit inverses."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    D = np.eye(len(a11)) - b11 @ a22
    X = solve(D, b11 @ a21)
    Y = solve(D, b12)
    return (a11 + a12 @ X, a12 @ Y,
            b21 @ (a21 + a22 @ X), b22 + b21 @ a22 @ Y)


def propagation(q, thickness, k0):
    P = np.diag(np.exp(1j * k0 * q * thickness))
    Z = np.zeros_like(P)
    return Z, P, P, Z


def rcwa(wavelength=0.633, period=0.8, layers=None, M=10,
         n_in=1.0, n_out=1.45, theta_deg=0.0, pol='TE'):
    """layers: list of (thickness, fill, eps_ridge, eps_gap), top to bottom.

    r,t refer to Ey (TE) or Z0*Hy (TM), incident amplitude = 1.
    Transmission reference planes are the top and bottom of the stack.
    """
    if layers is None:
        layers = [(0.35, 0.45, 2.0**2, 1.0)]
    if n_in <= 0 or n_out <= 0:
        raise ValueError('Positive real exterior indices required.')
    k0 = 2 * np.pi / wavelength
    orders = np.arange(-M, M + 1)
    kx = n_in * np.sin(np.deg2rad(theta_deg)) + orders * wavelength / period
    K = np.diag(kx)
    I = np.eye(len(kx))
    qi = outgoing_sqrt(n_in**2 - kx**2)
    qt = outgoing_sqrt(n_out**2 - kx**2)
    if min(np.min(np.abs(qi)), np.min(np.abs(qt))) < 1e-9:
        raise ValueError('Avoid exact exterior Rayleigh cutoff in this demo.')
    yi = qi if pol == 'TE' else qi / n_in**2
    yt = qt if pol == 'TE' else qt / n_out**2
    Wprev, Vprev = I, np.diag(yi)
    Z = np.zeros_like(I)
    S = (Z, I, I, Z)
    for d, fill, er, eg in layers:
        E = convolution(M, fill, er, eg)
        B = convolution(M, fill, 1 / er, 1 / eg)
        W, V, q = layer_modes(K, E, B, pol)
        S = star(S, interface(Wprev, Vprev, W, V))
        S = star(S, propagation(q, d, k0))
        Wprev, Vprev = W, V
    S = star(S, interface(Wprev, Vprev, I, np.diag(yt)))
    incident = np.zeros(len(kx), dtype=complex)
    incident[M] = 1
    r, t = S[0] @ incident, S[2] @ incident
    Rm = yi.real / yi[M].real * np.abs(r)**2
    Tm = yt.real / yi[M].real * np.abs(t)**2
    return dict(orders=orders, r=r, t=t, Rm=Rm, Tm=Tm,
                R=float(sum(Rm)), T=float(sum(Tm)),
                A=float(1 - sum(Rm) - sum(Tm)))


def fresnel_slab(wl, n0, n1, n2, d, theta, pol):
    kx = n0 * np.sin(np.deg2rad(theta))
    q = outgoing_sqrt(np.array([n0, n1, n2])**2 - kx**2)
    y = q if pol == 'TE' else q / np.array([n0, n1, n2])**2
    r01, r12 = (y[0] - y[1]) / (y[0] + y[1]), (y[1] - y[2]) / (y[1] + y[2])
    t01, t12 = 2 * y[0] / (y[0] + y[1]), 2 * y[1] / (y[1] + y[2])
    p = np.exp(2j * np.pi / wl * q[1] * d)
    den = 1 + r01 * r12 * p**2
    return (r01 + r12 * p**2) / den, t01 * t12 * p / den


def validate():
    errors = []
    for pol in ('TE', 'TM'):
        for theta in (0, 17, 41):
            for d in (0.0, 0.21, 1.8):
                result = rcwa(layers=[(d, 1.0, 2.1**2, 1)], M=4,
                              theta_deg=theta, pol=pol)
                rr, tt = fresnel_slab(0.633, 1, 2.1, 1.45, d, theta, pol)
                error = max(abs(result['r'][4] - rr), abs(result['t'][4] - tt))
                errors.append(float(error))
    assert max(errors) < 1e-10, errors
    bare = rcwa(layers=[], M=4)
    expected_R = ((1 - 1.45) / (1 + 1.45))**2
    assert abs(bare['R'] - expected_R) < 1e-12
    split_errors = []
    absorption = {}
    thick = {}
    symmetry = []
    for pol in ('TE', 'TM'):
        one = rcwa(pol=pol)
        split = rcwa(layers=[(0.175, .45, 4, 1)] * 2, pol=pol)
        split_errors.append(float(max(abs(one['r'] - split['r']))))
        assert split_errors[-1] < 1e-10
        symmetry.append(float(max(abs(one['Tm'] - one['Tm'][::-1]))))
        assert symmetry[-1] < 1e-10
        lossy = rcwa(layers=[(.35, .45, (2 + .08j)**2, 1)], pol=pol)
        assert 0 < lossy['A'] < 1
        absorption[pol] = lossy['A']
        deep = rcwa(layers=[(50.0, .45, 4, 1)], pol=pol)
        assert abs(deep['A']) < 1e-10
        thick[pol] = dict(R=deep['R'], T=deep['T'], residual=deep['A'])
    convergence = []
    reference = {pol: rcwa(M=50, pol=pol) for pol in ('TE', 'TM')}
    for M in (0, 1, 2, 3, 5, 8, 12, 18, 25, 35):
        row = dict(M=M, harmonics=2 * M + 1)
        for pol in ('TE', 'TM'):
            r = rcwa(M=M, pol=pol)
            assert abs(r['A']) < 1e-9
            row[pol] = dict(R=r['R'], T=r['T'], T0=float(r['Tm'][M]),
                            residual=r['A'],
                            difference_from_M50=abs(r['T'] - reference[pol]['T']))
        convergence.append(row)
    selected = {}
    for pol in ('TE', 'TM'):
        r = rcwa(M=35, pol=pol)
        selected[pol] = dict(R=r['R'], T=r['T'], A=r['A'],
                            orders=[dict(m=int(m), R=float(rr), T=float(tt))
                                    for m, rr, tt in zip(r['orders'], r['Rm'], r['Tm'])
                                    if abs(rr) + abs(tt) > 1e-12])
    return dict(units='um', example=dict(wavelength=.633, period=.8, thickness=.35,
                fill=.45, n_ridge=2, n_gap=1, n_in=1, n_out=1.45, theta_deg=0),
                max_slab_complex_amplitude_error=max(errors), slab_cases=len(errors),
                bare_interface_R=bare['R'], max_split_error=max(split_errors),
                max_symmetry_error=max(symmetry), absorption=absorption,
                thick_layer=thick, convergence=convergence, selected_M35=selected,
                reference_M50={p: dict(R=r['R'], T=r['T']) for p, r in reference.items()})


def figures(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': '#f8fafc', 'axes.facecolor': '#f8fafc',
        'text.color': '#153047', 'axes.labelcolor': '#153047',
        'savefig.facecolor': '#f8fafc', 'svg.fonttype': 'none'})
    output.mkdir(parents=True, exist_ok=True)
    blue, teal, orange = '#2563eb', '#0f9488', '#e57a25'

    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
        fig.savefig(output / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    def arrow(ax, start, end, color=blue, text=None):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2.4, color=color))
        if text:
            ax.text(*((np.array(start) + np.array(end)) / 2), text, fontsize=11,
                    color=color, ha='center', va='bottom')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(-1.8, 1.8), ylim=(1.5, -1.6))
    ax.axhspan(.5, 1.5, color='#d6e4f6')
    for x in np.arange(-2, 2, .8):
        ax.add_patch(Rectangle((x - .18, 0), .36, .5, color=teal, alpha=.8))
    ax.axhline(0, color='#789', lw=1); ax.axhline(.5, color='#789', lw=1)
    for x in (-.4, .4):
        ax.axvline(x, color='#8da0b0', ls='--', lw=1)
    arrow(ax, (-1.12, -1.3), (-1.12, 0), blue)
    ax.text(-1.7, -.7, '入射平面波', color=blue)
    for end, label in [((-.8, -1.25), 'm = −1'), ((.15, -1.25), 'm = 0'), ((1.2, -1.25), 'm = +1')]:
        arrow(ax, (.15, -.03), end, orange)
        ax.text(end[0], end[1] - .12, label, ha='center', color=orange)
    for end, label in [((-.75, 1.3), '−1'), ((.15, 1.3), '0'), ((1, 1.3), '+1')]:
        arrow(ax, (.15, .52), end, blue)
        ax.text(end[0], 1.44, label, ha='center', color=blue)
    ax.annotate('', xy=(-.4, -.25), xytext=(.4, -.25), arrowprops=dict(arrowstyle='<->', color=teal))
    ax.text(0, -.31, '周期 Λ', ha='center', color=teal)
    ax.text(1.75, -.6, '均匀入射区', ha='right')
    ax.text(1.75, .3, '周期图案层', ha='right')
    ax.text(1.75, 1.05, '均匀出射区', ha='right')
    arrow(ax, (-1.55, .8), (-1.1, .8), '#153047'); ax.text(-1.07, .8, 'x')
    arrow(ax, (-1.55, .8), (-1.55, 1.25), '#153047'); ax.text(-1.55, 1.4, 'z', ha='center')
    ax.axis('off')
    ax.set_title('RCWA：横向展开为谐波，纵向求解层内模态', fontsize=20, pad=18)
    fig.text(.5, .01, '几何与端口示意；箭头角度和长度不表示本算例的定量衍射角或功率。', ha='center', fontsize=10)
    save(fig, '01-geometry')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    x = np.linspace(-.5, .5, 1400)
    exact = np.where(abs(x) < .225, 4, 1)
    axs[0].plot(x, exact, color='#153047', lw=2, label='真实 ε(x)')
    for M, color in [(3, orange), (15, blue)]:
        m = np.arange(-M, M + 1)
        coeff = 3 * .45 * np.sinc(m * .45) + (m == 0)
        y = np.real(np.exp(2j * np.pi * x[:, None] * m) @ coeff)
        axs[0].plot(x, y, color=color, label=f'截断 M = {M}', lw=1.5)
    axs[0].set(xlabel='x / Λ', ylabel='相对介电常数', title='边界处的 Gibbs 振荡')
    axs[0].legend(fontsize=10)
    E = convolution(5, .45, 4, 1)
    im = axs[1].imshow(abs(E), cmap='Blues', origin='lower', extent=(-5.5, 5.5, -5.5, 5.5))
    axs[1].set(xlabel='输入谐波 n', ylabel='输出谐波 m', title=r'卷积矩阵 $|\varepsilon_{m-n}|$：非对角项产生耦合')
    fig.colorbar(im, ax=axs[1], shrink=.85)
    save(fig, '02-fourier-coupling')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    ax = axs[0]; ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis('off')
    ax.add_patch(FancyBboxPatch((3.2, 1), 3.6, 4, boxstyle='round,pad=.12', fc='#d6e4f6', ec=blue, lw=2))
    ax.text(5, 3, '层 / 多层结构\n散射矩阵 S', ha='center', va='center', fontsize=18)
    arrow(ax, (.3, 4.2), (3.05, 4.2), teal); ax.text(1.5, 4.65, 'aL  输入', ha='center')
    arrow(ax, (3.05, 1.8), (.3, 1.8), orange); ax.text(1.5, 1.05, 'bL  输出', ha='center')
    arrow(ax, (6.95, 4.2), (9.7, 4.2), orange); ax.text(8.5, 4.65, 'aR  输出', ha='center')
    arrow(ax, (9.7, 1.8), (6.95, 1.8), teal); ax.text(8.5, 1.05, 'bR  输入', ha='center')
    ax.set_title('端口约定：按输入与输出组织未知量')
    ad = np.linspace(0, 40, 300)
    axs[1].semilogy(ad, np.exp(ad), label='逆向跨层因子 exp(+αd)', color=orange)
    axs[1].semilogy(ad, np.exp(-ad), label='有界传播因子 exp(−αd)', color=blue)
    axs[1].set(xlabel='衰减常数 × 厚度  αd', ylabel='因子绝对值', title='倏逝模使朴素传输矩阵病态')
    axs[1].legend(fontsize=10); axs[1].grid(alpha=.2)
    save(fig, '03-scattering')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    rows = report['convergence']; ns = [r['harmonics'] for r in rows]
    for pol, color in [('TE', blue), ('TM', orange)]:
        axs[0].plot(ns, [r[pol]['T'] for r in rows], 'o-', color=color, label=pol)
        axs[1].semilogy(ns, [max(r[pol]['difference_from_M50'], 1e-16) for r in rows], 'o-', color=color, label=f'{pol}: |T − T(M=50)|')
        axs[1].semilogy(ns, [max(abs(r[pol]['residual']), 1e-16) for r in rows], ':', color=color, label=f'{pol}: |1 − R − T|')
    axs[0].set(xlabel='保留谐波数 N = 2M + 1', ylabel='总透射率 T', title='同一结构，增加谐波截断')
    axs[1].set(xlabel='保留谐波数 N = 2M + 1', ylabel='绝对差值', title='能量守恒并不保证结果已经收敛')
    for ax in axs:
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    save(fig, '04-convergence')

    fills = np.linspace(.08, .9, 120)
    values = [rcwa(wavelength=.633, period=.28, layers=[(.6, f, 2.4**2, 1)], M=18, pol='TE') for f in fills]
    t0 = np.array([v['t'][18] for v in values])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axs[0].plot(fills, np.unwrap(np.angle(t0)), color=blue)
    axs[0].set(xlabel='占空比 F', ylabel=r'展开后的 $\arg(t_0)$ / rad', title='真实 RCWA 算例：一维条纹的相位扫描')
    axs[1].plot(fills, [v['T'] for v in values], color=teal)
    axs[1].set(xlabel='占空比 F', ylabel=r'零级功率透射率 $T_0$', ylim=(0, 1.05), title='相位与透射率必须一起筛选')
    for ax in axs: ax.grid(alpha=.2)
    fig.supxlabel('教学参数：λ0=633 nm，Λ=280 nm，h=600 nm，n条纹=2.4，n基底=1.45，TE 正入射；非二维纳米柱库', fontsize=10)
    save(fig, '05-unit-cell-scan')
    library = [dict(fill=float(f), t_real=float(t.real), t_imag=float(t.imag),
                    T=float(v['T'])) for f, t, v in zip(fills, t0, values)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    r = np.linspace(-5, 5, 1000)
    phi = -2 * np.pi / .633 * (np.sqrt(10**2 + r**2) - 10)
    axs[0].plot(r, phi / (2*np.pi), color=blue, label='连续目标相位 / 2π')
    axs[0].plot(r, np.mod(phi, 2*np.pi) / (2*np.pi), color=orange, label='包裹到 [0, 2π) 后 / 2π')
    axs[0].set(xlabel='横向坐标 x / μm', ylabel='相位 / 2π', title='目标相位：λ0=633 nm，f=10 μm')
    axs[0].legend(fontsize=10); axs[0].grid(alpha=.2)
    ax = axs[1]; ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
    items = [('周期单元 RCWA', '复透射系数 · 偏振 · 色散'), ('目标相位匹配', '振幅约束 · 制造约束 · 候选筛选'), ('有限孔径传播', '局部周期近似 + 衍射传播'), ('器件验证', '邻域 / 超胞 / 全波 · 焦斑与效率')]
    for i, (title, sub) in enumerate(items):
        y = 8.0 - i*2.2
        ax.add_patch(FancyBboxPatch((.45, y), 9.1, 1.5, boxstyle='round,pad=.13', fc='#e5f2f0' if i%2 == 0 else '#e5edf9', ec='none'))
        ax.text(5, y+.95, title, ha='center', va='center', fontsize=15, weight='bold')
        ax.text(5, y+.38, sub, ha='center', va='center', fontsize=10)
        if i < 3: arrow(ax, (5, y-.05), (5, y-.63), teal)
    save(fig, '06-metalens-workflow')
    return library


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--figures-dir', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = validate()
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.figures_dir:
        library = figures(args.figures_dir, report)
        (args.output_dir / 'unit_cell_scan.json').write_text(json.dumps(library, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('slab_cases', 'max_slab_complex_amplitude_error',
                     'max_split_error', 'max_symmetry_error', 'selected_M35')}, indent=2))
