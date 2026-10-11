"""Reproduce the Wallis article's six figures and independent numerical checks.

Run: python source/downloads/wallis-formula/figures-and-checks.py
Requires: numpy, scipy, matplotlib. Chinese labels use Microsoft YaHei.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'img' / 'wallis-formula'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, INK = '#2262a8', '#128475', '#d66a32', '#23374d'
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': '#f4f7fb',
    'axes.facecolor': 'white', 'savefig.facecolor': '#f4f7fb'})

def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)

def integral(n):
    return math.exp(betaln((n + 1) / 2, .5)) / 2

# 1. A reading map, also used as the article card cover.
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set(xlim=(0, 12), ylim=(0, 6.5)); ax.axis('off')
ax.text(.3, 5.8, '华里士公式：面积怎样变成无穷乘积', fontsize=23, color=INK, weight='bold')
ax.text(.3, 5.22, '一族积分 · 两条递推链 · 一次夹逼', fontsize=15, color=GREEN)
boxes = [(0.3, '① 面积', r'$I_n=\int_0^{\pi/2}\sin^n x\,dx$', BLUE),
         (4.35, '② 递推', r'$I_n=\frac{n-1}{n}I_{n-2}$', GREEN),
         (8.4, '③ 夹逼', r'$I_{n+1}/I_n\ \to\ 1$', ORANGE)]
for xx, label, eq, color in boxes:
    ax.add_patch(FancyBboxPatch((xx, 3.05), 3.25, 1.5, boxstyle='round,pad=.12', fc='white', ec=color, lw=2))
    ax.text(xx+.2, 4.1, label, color=color, fontsize=17)
    ax.text(xx+1.62, 3.48, eq, ha='center', fontsize=17)
for xx in [3.65, 7.7]:
    ax.annotate('', xy=(xx+.5, 3.8), xytext=(xx, 3.8), arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 2})
ax.text(6, 1.82, r'$\frac{\pi}{2}=\prod_{k=1}^{\infty}\frac{4k^2}{4k^2-1}$', ha='center', fontsize=34, color=INK)
ax.text(6, .63, '每一项都大于 1，有限乘积却始终小于 π/2。为什么？', ha='center', fontsize=16, color=INK)
save(fig, '01-roadmap.png')

# 2. Function ordering and actual areas.
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
x = np.linspace(0, np.pi/2, 900)
for n, c in [(0, '#8c94a3'), (1, BLUE), (2, GREEN), (5, ORANGE), (20, '#8251aa')]:
    axs[0].plot(x, np.sin(x)**n, label=f'n = {n}', color=c, lw=2.2)
axs[0].set(title='指数增加，曲线向右端收缩', xlabel='x', ylabel=r'$\sin^n x$')
axs[0].legend(fontsize=10)
for n, c in [(4, BLUE), (5, ORANGE), (6, GREEN)]:
    y = np.sin(x)**n
    axs[1].fill_between(x, 0, y, color=c, alpha=.09)
    axs[1].plot(x, y, color=c, lw=2.4, label=fr'$I_{n}={integral(n):.5f}$')
axs[1].set(title=r'面积严格排序：$I_6<I_5<I_4$', xlabel='x', ylabel='曲线高度')
axs[1].legend(loc='upper left')
for ax in axs:
    ax.set_xticks([0, np.pi/4, np.pi/2], ['0', r'$\pi/4$', r'$\pi/2$']); ax.grid(alpha=.16)
save(fig, '02-sine-areas.png')

# 3. Exact ratio between computable lower and upper bounds.
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
n = np.arange(1, 41)
ratio = np.array([integral(2*k+1)/integral(2*k) for k in n])
axs[0].fill_between(n, 2*n/(2*n+1), 1, color=GREEN, alpha=.09)
axs[0].plot(n, 2*n/(2*n+1), '--', color=GREEN, label=r'下界 $2n/(2n+1)$')
axs[0].plot(n, ratio, 'o-', ms=3, color=BLUE, label=r'$I_{2n+1}/I_{2n}$')
axs[0].axhline(1, color=ORANGE, label='上界 1')
axs[0].set(xlabel='n', ylabel='比值', title='夹住比值，才能控制相对误差'); axs[0].legend(fontsize=10)
nn = np.arange(1, 301)
iv = np.array([integral(int(k)) for k in nn])
axs[1].loglog(nn, iv, color=BLUE, lw=2.5, label=r'$I_n$')
axs[1].loglog(nn, np.sqrt(np.pi/(2*nn)), '--', color=ORANGE, label=r'$\sqrt{\pi/(2n)}$')
axs[1].set(xlabel='n（对数坐标）', ylabel='积分值（对数坐标）', title='积分趋于 0，但相邻项的比趋于 1'); axs[1].legend()
for ax in axs: ax.grid(alpha=.16)
save(fig, '03-squeeze.png')

# 4. Finite-product bracket and error scaling.
N = 10000
ks = np.arange(1, N+1, dtype=float)
products = np.exp(np.cumsum(-np.log1p(-1/(4*ks**2))))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
nn = ks[:40]; pp = products[:40]
axs[0].fill_between(nn, 2*pp, 2*pp*(2*nn+1)/(2*nn), alpha=.12, color=BLUE)
axs[0].plot(nn, 2*pp, color=BLUE, label=r'下界 $2P_n$')
axs[0].plot(nn, 2*pp*(2*nn+1)/(2*nn), color=ORANGE, label=r'上界 $2P_n(1+1/(2n))$')
axs[0].axhline(np.pi, ls='--', color=INK, label=r'$\pi$')
axs[0].set(xlabel='n', ylabel='π 的上下界', title='每个有限 n 都给出一个可信区间'); axs[0].legend(fontsize=10)
axs[1].loglog(ks, np.pi-2*products, color=BLUE, lw=2.5, label=r'实际误差 $\pi-2P_n$')
axs[1].loglog(ks, np.pi/(4*ks), '--', color=ORANGE, label=r'渐近值 $\pi/(4n)$')
axs[1].loglog(ks, np.pi/(2*ks+1), ':', color=GREEN, label=r'严格上界 $\pi/(2n+1)$')
axs[1].set(xlabel='n（对数坐标）', ylabel='绝对误差（对数坐标）', title='收敛速度约为 1/n'); axs[1].legend(fontsize=10)
for ax in axs: ax.grid(alpha=.16)
save(fig, '04-product-error.png')

# 5. Rescale the endpoint boundary layer; this is not a fit.
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
t = np.linspace(0, 1.25, 700); u = np.linspace(0, 3.7, 700)
for n, c in [(4, ORANGE), (16, GREEN), (64, BLUE)]:
    axs[0].plot(t, np.cos(t)**n, color=c, label=f'n = {n}', lw=2.3)
    z = u/np.sqrt(n)
    val = np.where(z <= np.pi/2, np.maximum(np.cos(z), 0)**n, 0)
    axs[1].plot(u, val, color=c, lw=2.3, label=f'n = {n}')
axs[0].set(xlabel=r'$t=\pi/2-x$', ylabel=r'$\cos^n t$', title='峰高为 1，峰宽随 n 增大而缩小')
axs[1].plot(u, np.exp(-u*u/2), '--', color=INK, lw=2, label=r'$e^{-u^2/2}$')
axs[1].set(xlabel=r'$u=\sqrt{n}\,t$', ylabel='缩放后的曲线高度', title='横轴放大 √n 倍后，显现半高斯形状')
for ax in axs: ax.grid(alpha=.16); ax.legend(fontsize=10)
save(fig, '05-gaussian-limit.png')

# 6. The central binomial coefficient as a fair-coin probability.
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
nn = np.arange(1, 81)
prob = np.array([math.comb(2*int(n), int(n))/4**int(n) for n in nn])
axs[0].plot(nn, prob, color=BLUE, lw=2.5, label=r'精确值 $\binom{2n}{n}/4^n$')
axs[0].plot(nn, 1/np.sqrt(np.pi*nn), '--', color=ORANGE, label=r'近似值 $1/\sqrt{\pi n}$')
axs[0].set(xlabel='n（抛硬币次数为 2n）', ylabel='正反面各半的概率', title='组合计数中出现 π'); axs[0].legend(fontsize=10)
axs[1].fill_between(nn, np.sqrt(2*nn/(2*nn+1)), 1, alpha=.12, color=GREEN)
axs[1].plot(nn, prob*np.sqrt(np.pi*nn), color=BLUE, lw=2.5, label=r'$\sqrt{\pi n}\,\binom{2n}{n}/4^n$')
axs[1].plot(nn, np.sqrt(2*nn/(2*nn+1)), '--', color=GREEN, label=r'下界 $\sqrt{2n/(2n+1)}$')
axs[1].axhline(1, color=ORANGE, label='上界 1')
axs[1].set(xlabel='n', ylabel='精确值 / 渐近值', title='无需 Stirling 公式也能证明渐近式'); axs[1].legend(fontsize=10)
for ax in axs: ax.grid(alpha=.16)
save(fig, '06-binomial.png')

# Independent quadrature against recurrence, including real exponents near -1.
checks = []
rec = [math.pi/2, 1.0]
for n in range(2, 81): rec.append((n-1)/n*rec[n-2])
for n in list(range(31)) + [50, 80, -.9, -.5, .5, 2.3]:
    q, _ = quad(lambda x: np.sin(x)**n, 0, np.pi/2, epsabs=2e-11, epsrel=2e-11)
    exact = rec[n] if isinstance(n, int) and n >= 0 else integral(n)
    rel = abs(q/exact-1)
    assert rel < 2e-9, (n, rel)
    checks.append({'n': n, 'quadrature': q, 'closed_or_recurrence': exact, 'relative_error': rel})
for n in range(1, 80):
    assert n/(n+1) < rec[n+1]/rec[n] < 1
    assert abs(rec[n]*rec[n+1] / (math.pi/(2*(n+1)))-1) < 2e-14
assert np.all(2*products < math.pi)
assert np.all(math.pi < 2*products*(2*ks+1)/(2*ks))
assert np.all(np.diff(products) > 0)
assert np.all(np.diff(products*(2*ks+1)/(2*ks)) < 0)
assert np.all(1/np.sqrt(np.pi*(nn+.5)) < prob)
assert np.all(prob < 1/np.sqrt(np.pi*nn))
table = [{'n': n, 'two_Pn': float(2*products[n-1]),
          'absolute_error': float(math.pi-2*products[n-1]),
          'relative_error': float(1-2*products[n-1]/math.pi)} for n in [1, 2, 5, 10, 100, 1000, 10000]]
report = {'quadrature_checks': checks, 'max_quadrature_relative_error': max(c['relative_error'] for c in checks),
          'product_brackets_checked': N, 'adjacent_identities_checked': 79,
          'binomial_brackets_checked': 80, 'table': table,
          'n_times_pi_error_at_10000': float(N*(math.pi-2*products[-1])),
          'asymptotic_target_pi_over_4': math.pi/4,
          'figures': sorted(p.name for p in OUT.glob('*.png'))}
(HERE/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'quadrature_checks'}, ensure_ascii=False, indent=2))
