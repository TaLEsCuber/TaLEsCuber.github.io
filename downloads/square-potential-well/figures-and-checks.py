"""Reproduce the square-well tutorial figures and independent numerical checks.

Run: python source/downloads/square-potential-well/figures-and-checks.py
Dependencies: numpy, scipy, matplotlib. No dimensional parameters are implicit:
y=x/a, Ea=hbar**2/(2*m*a**2), rho=sqrt(V0/Ea).
Default outputs are resolved relative to this script's repository location.
Use --output-dir DIR to render elsewhere after downloading this file alone.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
ROOT = BASE.parents[2]
OUT = args.output_dir or ROOT / 'source/img/square-potential-well'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'font.size': 11, 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.labelcolor': '#23344d',
    'text.color': '#23344d', 'xtick.color': '#526278',
    'ytick.color': '#526278', 'figure.facecolor': '#faf9f5',
    'axes.facecolor': '#faf9f5', 'svg.fonttype': 'none',
    'lines.linewidth': 2.2, 'savefig.facecolor': '#faf9f5',
})
TEAL, ORANGE, BLUE, RED = '#087f8c', '#c27822', '#496cba', '#ac4561'
COLORS = [TEAL, ORANGE, BLUE, RED]


def levels(rho):
    """Use a monotone quantization function; never cross a tan/cot pole."""
    result = []
    for n in range(1, int(np.ceil(2*rho/np.pi)) + 2):
        target = n*np.pi/2
        if target >= rho + np.pi/2 - 1e-13:
            continue  # zero energy is not square integrable
        z = brentq(lambda u: u + np.arcsin(u/rho) - target,
                   0, rho, xtol=5e-15)
        w = np.sqrt((rho-z)*(rho+z))
        result.append({'n': n, 'parity': 'even' if n % 2 else 'odd',
                       'z': z, 'w': w, 'energy_Ea': -w*w})
    return result


def wave(y, lev):
    y = np.asarray(y)
    z, w = lev['z'], lev['w']
    amp = (1+1/w)**-.5  # sqrt(a)*A
    if lev['parity'] == 'even':
        return amp*np.where(abs(y) <= 1, np.cos(z*y),
                            np.cos(z)*np.exp(-w*(abs(y)-1)))
    return amp*np.where(abs(y) <= 1, np.sin(z*y),
                        np.sign(y)*np.sin(z)*np.exp(-w*(abs(y)-1)))


def amplitudes(s, rho):
    s = np.asarray(s, dtype=complex)
    q = np.sqrt(s*s+rho*rho)
    # sin(2q)/q = 2*sinc(2q/pi) also works at q=0.
    h = 2*np.sinc(2*q/np.pi)
    f = 2*s*np.cos(2*q)-1j*(s*s+q*q)*h
    t = 2*s*np.exp(-2j*s)/f
    r = 1j*(q*q-s*s)*h*np.exp(-2j*s)/f
    return t, r


def parity_s(s, rho):
    q = np.sqrt(s*s+rho*rho+0j)
    c, sn = np.cos(q), np.sin(q)
    se = np.exp(-2j*s)*(s*c+1j*q*sn)/(s*c-1j*q*sn)
    so = np.exp(-2j*s)*(q*c+1j*s*sn)/(q*c-1j*s*sn)
    return se, so


def scattering_wave(y, s, rho):
    y = np.asarray(y)
    q = np.sqrt(s*s+rho*rho)
    t, r = amplitudes(s, rho)
    inside = t*np.exp(1j*s)*(np.cos(q*(y-1))
                            + 1j*s/q*np.sin(q*(y-1)))
    return np.where(y < -1, np.exp(1j*s*y)+r*np.exp(-1j*s*y),
                    np.where(y > 1, t*np.exp(1j*s*y), inside))


def save(fig, stem):
    fig.savefig(OUT / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)


def shade(ax):
    ax.axvspan(-1, 1, color=TEAL, alpha=.06)
    for x in [-1, 1]:
        ax.axvline(x, color='#8693a3', lw=1, ls='--')


rho = 4.5
lv = levels(rho)
fig, ax = plt.subplots(figsize=(10.6, 5.3), layout='constrained')
ax.plot([-3, -1, -1, 1, 1, 3], [0, 0, -rho*rho, -rho*rho, 0, 0], color='#23344d')
for lev, col in zip(lv, COLORS):
    e = lev['energy_Ea']
    ax.hlines(e, -1, 1, color=col, lw=2.6)
    ax.text(1.12, e, f"n={lev['n']}，{'偶' if lev['n']%2 else '奇'}宇称", color=col, va='center')
ax.axhline(4, color=RED, ls='--')
ax.text(-2.8, 4.6, 'E > 0：散射态，外部也可传播', color=RED)
ax.text(-2.85, -7, '−V0 < E < 0\n离散束缚能级', linespacing=1.8)
ax.text(0, -rho*rho-1.9, '宽度 L = 2a', ha='center')
ax.set(xlim=(-3, 3), ylim=(-24, 8), xlabel='位置 x/a', ylabel='能量 / Ea',
       title='同一个方势阱：离散束缚谱与连续散射谱（ρ = 4.5）')
save(fig, '01-spectrum')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
for ax, par, title in zip(axes, ['even', 'odd'], ['偶宇称：z tan z = w', '奇宇称：−z cot z = w']):
    z = np.linspace(.0001, rho-.0001, 7000)
    yy = z*np.tan(z) if par == 'even' else -z/np.tan(z)
    yy[(yy < 0) | (yy > rho+.3)] = np.nan
    ax.plot(z, yy, color=TEAL, label='边界匹配曲线')
    ax.plot(z, np.sqrt(rho*rho-z*z), color=ORANGE, label='圆：z²+w²=ρ²')
    for lev in lv:
        if lev['parity'] == par:
            ax.scatter(lev['z'], lev['w'], color=RED, zorder=5)
            ax.annotate(f"n={lev['n']}", (lev['z'], lev['w']), xytext=(8, 9), textcoords='offset points')
    ax.set(xlim=(0, rho+.12), ylim=(0, rho+.3), xlabel='z = qa', ylabel='w = κa', title=title)
    ax.legend(loc='lower left', fontsize=9)
save(fig, '02-quantization')

fig, axes = plt.subplots(3, 1, figsize=(10.6, 8), layout='constrained', sharex=True)
y = np.linspace(-3.4, 3.4, 2500)
for ax, lev, col in zip(axes, lv, COLORS):
    shade(ax)
    psi = wave(y, lev)
    ax.plot(y, psi, color=col, label='归一化波函数 √a ψ')
    ax.fill_between(y, 0, psi, where=abs(y)>1, color=col, alpha=.2)
    ax.axhline(0, color='#bac1c9', lw=.8)
    ax.set(ylabel='√a ψ(x)', title=f"n={lev['n']}，E/Ea={lev['energy_Ea']:.6f}，节点数={lev['n']-1}")
    ax.text(.98, .82, f"衰减长度 ℓ/a = {1/lev['w']:.3f}", ha='right', transform=ax.transAxes)
axes[-1].set_xlabel('位置 x/a；阴影外侧是指数尾部')
save(fig, '03-bound-waves')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
rs = np.geomspace(.025, 1.0, 180)
eb = np.array([-levels(r)[0]['energy_Ea'] for r in rs])
axes[0].loglog(rs, eb, color=TEAL, label='精确束缚能 |E|/Ea')
axes[0].loglog(rs, rs**4, '--', color=ORANGE, label='浅阱首项 ρ⁴')
axes[0].set(xlabel='势阱强度 ρ', ylabel='束缚能 |E|/Ea', title='浅阱束缚能按 V0² 消失')
axes[0].legend(fontsize=9)
y = np.linspace(-35, 35, 2500)
for r, col in zip([.25, .5, .8], COLORS):
    axes[1].plot(y, wave(y, levels(r)[0])**2, color=col, label=f'ρ={r}')
shade(axes[1])
axes[1].set(xlabel='位置 x/a', ylabel='概率密度 a|ψ|²', title='越浅越向阱外扩展')
axes[1].legend()
save(fig, '04-shallow-limit')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
rs = np.geomspace(4.8, 100, 160)
for n, col in zip([1, 2, 3], COLORS):
    exact = np.array([levels(r)[n-1]['z']**2 for r in rs])
    en_inf = (n*np.pi/2)**2
    axes[0].semilogx(rs, exact/en_inf, color=col, label=f'n={n}')
    axes[1].loglog(rs, 1-exact/en_inf, color=col, label=f'n={n}')
axes[0].axhline(1, color='#8693a3', ls='--', lw=1)
axes[1].loglog(rs, 2/rs, '--', color=RED, label='固定 n 渐近线 2/ρ')
axes[0].set(xlabel='势阱强度 ρ', ylabel='(En+V0) / εn(∞)', title='比较相对阱底的能量')
axes[1].set(xlabel='势阱强度 ρ', ylabel='相对无限深阱的能量降低', title='固定低能级的深阱修正')
for ax in axes: ax.legend(fontsize=9)
save(fig, '05-deep-limit')

fig, axes = plt.subplots(2, 1, figsize=(10.6, 8), layout='constrained')
es = np.linspace(.001, 70, 5000)
tt = abs(amplitudes(np.sqrt(es), rho)[0])**2
axes[0].plot(es, tt, color=TEAL)
res = [(n*np.pi/2)**2-rho*rho for n in range(1, 7)]
for e in res:
    if 0 < e < 70:
        axes[0].scatter(e, 1, color=ORANGE, zorder=5)
axes[0].set(xlabel='入射能量 E/Ea', ylabel='透射率 T', ylim=(0, 1.08),
            title='正能量完全透射：2qa = jπ（ρ = 4.5）')
y = np.linspace(-4.2, 4.2, 2600)
for e, col, label in [(res[2], TEAL, '共振'), (.3, ORANGE, '非共振')]:
    psi = scattering_wave(y, np.sqrt(e), rho)
    axes[1].plot(y, abs(psi)**2, color=col,
                 label=f'{label}：E/Ea={e:.4f}，T={abs(amplitudes(np.sqrt(e),rho)[0])**2:.3f}')
shade(axes[1])
axes[1].set(xlabel='位置 x/a', ylabel='|ψ|² / 入射波密度', title='入射归一化的散射密度：左侧可有干涉条纹')
axes[1].legend(fontsize=9, loc='upper right')
save(fig, '06-scattering')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
rs = np.linspace(1.35, 1.80, 1800)
for s, col in zip([.015, .04, .12], COLORS):
    axes[0].plot(rs, abs(amplitudes(s, rs)[0])**2, color=col, label=f'ka={s}')
axes[0].axvline(np.pi/2, color='#8693a3', ls='--', lw=1)
axes[0].set(xlabel='势阱强度 ρ', ylabel='透射率 T', ylim=(0, 1.08), title='新奇束缚态的阈值：ρ = π/2')
axes[0].legend(fontsize=9)
ss = np.geomspace(.0001, .5, 500)
for r, col, label in [(1.48, ORANGE, '阈值前'), (np.pi/2, TEAL, '恰在阈值'), (1.67, BLUE, '阈值后')]:
    axes[1].semilogx(ss, abs(amplitudes(ss,r)[0])**2, color=col, label=f'{label}，ρ={r:.4f}')
axes[1].set(xlabel='入射波数 ka', ylabel='透射率 T', ylim=(0, 1.08), title='零能极限取决于是否恰在阈值')
axes[1].legend(fontsize=9)
save(fig, '07-threshold')

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), layout='constrained')
ax = axes[0]
ax.axhline(0, color='#8693a3', lw=1)
ax.axvline(0, color='#8693a3', lw=1)
ax.plot([-2.5, 2.5], [0, 0], color=TEAL, lw=3)
for lev, col in zip(lv, COLORS):
    ax.scatter(0, lev['w'], color=col, s=65, zorder=5)
    ax.text(.16, lev['w'], f"n={lev['n']}，ka=i×{lev['w']:.3f}", va='center', fontsize=10)
ax.text(-2.25, .20, '实轴：E > 0 散射态', color=TEAL, fontsize=10)
ax.text(-2.25, -1.05, '下半平面还可能有虚态或共振极点\n本图只标出 ρ=4.5 的束缚极点', fontsize=9, linespacing=1.6)
ax.set(xlim=(-2.5, 3.4), ylim=(-1.6, 4.9), xlabel='Re(ka)', ylabel='Im(ka)',
       title='束缚态位于正虚轴：k = iκ')
ss = np.geomspace(1e-4, 1500, 45000)
se, so = parity_s(ss, rho)
de, do = np.unwrap(np.angle(se))/2, np.unwrap(np.angle(so))/2
de -= round(de[-1]/np.pi)*np.pi
do -= round(do[-1]/np.pi)*np.pi
axes[1].semilogx(ss, de/np.pi, color=TEAL, label='偶宇称 δe/π')
axes[1].semilogx(ss, do/np.pi, color=ORANGE, label='奇宇称 δo/π')
axes[1].axhline(1.5, color=TEAL, ls=':', lw=1)
axes[1].axhline(1.0, color=ORANGE, ls=':', lw=1)
axes[1].set(xlim=(1e-4, 100), xlabel='波数 ka（对数坐标）', ylabel='从高能连续展开的相移 / π',
            title='低能相移读出束缚态数：Ne=2，No=1')
axes[1].legend(fontsize=9)
save(fig, '08-poles-phases')

# Independent validation: direct boundary matrix, numerical quadrature,
# a real-space Hamiltonian, and parity channels versus traveling waves.
report = {'units': 'x/a and Ea=hbar^2/(2ma^2)', 'rho': rho,
          'levels': lv, 'figures': 8}
norm_err, match_err, pole_err = [], [], []
for lev in lv:
    z, w = lev['z'], lev['w']
    integral = 2*quad(lambda yy: float(wave(yy,lev))**2, 0, 1, epsabs=1e-12)[0]
    integral += 2*quad(lambda yy: float(wave(yy,lev))**2, 1, np.inf, epsabs=1e-12)[0]
    norm_err.append(abs(integral-1))
    match_err.append(abs(z*np.tan(z)-w) if lev['parity']=='even' else abs(-z/np.tan(z)-w))
    s = 1j*w
    h = 2*np.sinc(2*z/np.pi)
    f = 2*s*np.cos(2*z)-1j*(s*s+z*z)*h
    pole_err.append(abs(f))
    amp2 = 1/(1+1/w)
    b2 = np.cos(z)**2 if lev['parity']=='even' else np.sin(z)**2
    lev['outside_probability'] = amp2*b2/w
report['normalization_max_error'] = max(norm_err)
report['bound_matching_max_error'] = max(match_err)
report['pole_denominator_max_abs'] = max(pole_err)
travel_err, parity_err, flux_err = [], [], []
for e in [.00001, .1, 1., 3., 10., 30., 100.]:
    s, q = np.sqrt(e), np.sqrt(e+rho*rho)
    # Unknown vector [r,A,B,t]; independent four-equation interface solve.
    mat = np.array([
        [np.exp(1j*s), -np.exp(-1j*q), -np.exp(1j*q), 0],
        [-s*np.exp(1j*s), -q*np.exp(-1j*q), q*np.exp(1j*q), 0],
        [0, np.exp(1j*q), np.exp(-1j*q), -np.exp(1j*s)],
        [0, q*np.exp(1j*q), -q*np.exp(-1j*q), -s*np.exp(1j*s)],
    ], dtype=complex)
    rhs = np.array([-np.exp(-1j*s), -s*np.exp(-1j*s), 0, 0])
    rr, aa, bb, tt = np.linalg.solve(mat, rhs)
    t, r = amplitudes(s,rho)
    se, so = parity_s(s,rho)
    travel_err.append(max(abs(tt-t),abs(rr-r)))
    parity_err.append(max(abs((se+so)/2-t),abs((se-so)/2-r)))
    flux_err.append(abs(abs(t)**2+abs(r)**2-1))
report['boundary_matrix_max_error'] = max(travel_err)
report['parity_amplitudes_max_error'] = max(parity_err)
report['flux_max_error'] = max(flux_err)
report['shallow_tests'] = []
for r in [.025,.05,.1,.2]:
    e = -levels(r)[0]['energy_Ea']
    report['shallow_tests'].append({'rho':r,'binding_Ea':e,'ratio_to_rho4':e/r**4,
                                  'next_order_ratio':1-4*r*r/3})
report['count_at_thresholds'] = {str(r):len(levels(r)) for r in [.1,np.pi/2,np.pi/2+1e-6,np.pi,4.5]}
report['low_energy_T'] = {str(r):float(abs(amplitudes(1e-5,r)[0])**2) for r in [1.48,np.pi/2,1.67]}
report['levinson_delta_over_pi'] = {'even':float(de[0]/np.pi),'odd':float(do[0]/np.pi)}
report['deep_limit_tests'] = []
for r in [20., 50., 100.]:
    en = levels(r)[0]['z']**2
    deficit = 1-en/(np.pi/2)**2
    report['deep_limit_tests'].append({'rho':r,'ground_epsilon_Ea':en,
                                     'deficit_over_2_by_rho':deficit/(2/r)})
report['threshold_binding_tests'] = []
for dr in [.01,.003,.001]:
    rho_c = np.pi/2
    en = -levels(rho_c+dr)[1]['energy_Ea']
    report['threshold_binding_tests'].append({'delta_rho':dr,'binding_Ea':en,
        'ratio_to_quadratic_law':en/(rho_c*dr)**2})
report['finite_difference'] = []
for spacing in [.004,.002]:
    # Cell-centred grid: interfaces are halfway between grid points.
    num = round(16/spacing)
    y = -8 + (np.arange(num)+.5)*spacing
    pot = np.where(abs(y)<1,-rho*rho,0.)
    vals = eigh_tridiagonal(2/spacing**2+pot, -np.ones(num-1)/spacing**2,
                            select='v',select_range=(-rho*rho,-1e-6),eigvals_only=True)
    report['finite_difference'].append({'spacing_a':spacing,'energies_Ea':vals.tolist(),
        'max_abs_error_Ea':float(max(abs(vals-np.array([l['energy_Ea'] for l in lv]))))})
assert report['normalization_max_error'] < 1e-10
assert report['bound_matching_max_error'] < 1e-10
assert report['boundary_matrix_max_error'] < 1e-10
assert report['parity_amplitudes_max_error'] < 1e-10
assert report['flux_max_error'] < 1e-10
assert report['finite_difference'][-1]['max_abs_error_Ea'] < report['finite_difference'][0]['max_abs_error_Ea']
assert report['count_at_thresholds'][str(np.pi/2)] == 1
assert report['low_energy_T'][str(np.pi/2)] > .9999
assert report['low_energy_T']['1.48'] < 1e-7
assert report['low_energy_T']['1.67'] < 1e-7
assert abs(de[0]/np.pi-1.5) < 1e-4 and abs(do[0]/np.pi-1) < 1e-4
assert abs(report['deep_limit_tests'][-1]['deficit_over_2_by_rho']-1) < .02
assert abs(report['threshold_binding_tests'][-1]['ratio_to_quadratic_law']-1) < .003
assert abs(report['shallow_tests'][0]['ratio_to_rho4']-1) < .001
target = OUT/'validation.json' if args.output_dir else BASE/'validation.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
