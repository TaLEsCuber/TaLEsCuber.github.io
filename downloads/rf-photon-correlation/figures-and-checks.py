"""Original tutorial calculations. All plotted data are synthetic, never lab data.

Run: python source/downloads/rf-photon-correlation/figures-and-checks.py
Requires numpy, scipy, matplotlib. Gamma=1 in the response calculation.
The AI illustrations and the user's reference are not modified by this script.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'img' / 'rf-photon-correlation'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 11,
    'figure.facecolor': '#faf9f5', 'axes.facecolor': '#faf9f5',
    'axes.spines.top': False, 'axes.spines.right': False})
TEAL, VIOLET, GRAY, RED = '#00868b', '#8054bd', '#6d7684', '#ca6946'

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)

def steady(delta, s):
    laser = np.sqrt(s / 2)
    J = np.array([[-.5, -delta, 0.], [delta, -.5, -laser], [0., laser, -1.]])
    state = np.linalg.solve(J, [0., 0., 1.])
    return J, state

def transfer(delta, s, omega):
    """C/beta for delta(t)=delta-beta*omega*cos(theta), C=X-iY."""
    J, state = steady(delta, s)
    force = -omega * np.array([-state[1], state[0], 0.])
    response = np.linalg.solve(1j * omega * np.eye(3) - J, force)
    return response[2] / (1 + state[2])

def estimate(counts, theta):
    return 2 * np.sum(counts * np.cos(theta)) / counts.sum(), 2 * np.sum(counts * np.sin(theta)) / counts.sum()

checks = {}
for delta in [-2., -.5, 0., .7]:
    for s in [.02, .2, 1.]:
        _, state = steady(delta, s)
        expected = s / (2 * (1 + s + 4 * delta**2))
        assert np.isclose((1 + state[2]) / 2, expected, rtol=1e-10)
checks['steady_state_lorentzian'] = 'passed'

small_omega = 1e-5
expected_static = 8 * (-.5) * small_omega / (1 + .2 + 4 * .5**2)
static_error = abs(transfer(-.5, .2, small_omega) / expected_static - 1)
assert static_error < 1e-4
checks['adiabatic_complex_relative_error'] = float(static_error)

# Independent weak-drive optical-carrier/sideband response check, magnitude only.
def sideband_slope(delta, omega):
    A = lambda d: 1 / (.5 + 1j * d)
    a0, ap, am = A(delta), A(delta + omega), A(delta - omega)
    return abs(np.conj(a0) * ap - a0 * np.conj(am)) / abs(a0)**2

weak_errors = []
for delta in [-2., -1., -.5, -.1]:
    for omega in [.1, 1., 2.]:
        weak_errors.append(abs(abs(transfer(delta, 1e-5, omega)) / sideband_slope(delta, omega) - 1))
assert max(weak_errors) < 3e-5
checks['weak_drive_sideband_max_relative_error'] = float(max(weak_errors))

# Verify finite-frequency linear response against time-domain nonlinear Bloch ODE.
delta, s, omega, beta = -.7, .2, 1.3, 1e-4
J, state = steady(delta, s)
laser = np.sqrt(s / 2)
def rhs(t, y):
    d = delta - beta * omega * np.cos(omega * t)
    u, v, w = y
    return [-u / 2 - d * v, d * u - v / 2 - laser * w, laser * v - w - 1]
period = 2 * np.pi / omega
times = np.linspace(0, 80 * period, 80 * 256 + 1)
sol = solve_ivp(rhs, [times[0], times[-1]], state, t_eval=times, rtol=1e-10, atol=1e-12)
sel = (times >= 40 * period) & (times < 80 * period)
rho = (sol.y[2, sel] + 1) / 2
design = np.column_stack([np.ones(sel.sum()), np.cos(omega * times[sel]), np.sin(omega * times[sel])])
fit = np.linalg.lstsq(design, rho, rcond=None)[0]
measured = (fit[1] - 1j * fit[2]) / fit[0] / beta
error = abs(measured / transfer(delta, s, omega) - 1)
assert error < 1e-4
checks['bloch_time_domain_relative_error'] = float(error)

# Fig: slow secular motion with intrinsic RF ripple vs displaced equilibrium EMM.
t = np.linspace(0, 4, 10000)
q, secular, offset = .24, .4, 1.1
slow = secular * np.cos(2 * np.pi * t)
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout='constrained')
for ax, center, title in zip(axes, [0, offset], ['均值位于 RF 零点：仍可有固有微运动', '静态杂散场推离 RF 零点：增加过量微运动']):
    track = (center + slow) * (1 - q / 2 * np.cos(2 * np.pi * 25 * t))
    ax.plot(t, track, color=VIOLET, lw=1.2, label='总运动（小 q 近似）')
    ax.plot(t, center + slow, color=TEAL, lw=1.6, label='慢运动包络中心')
    ax.axhline(0, color=GRAY, ls=':', label='RF 零点')
    ax.set(title=title, ylabel='位移 / 任意单位')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
axes[-1].set_xlabel('时间 / 世俗运动周期')
fig.suptitle('微运动从哪里来？  |  理想一维模型，非实际阱轨迹', fontsize=15)
save(fig, 'motion-components')

# Fig: synthetic phase histograms and nonzero quadrature scan floor.
rng = np.random.default_rng(20260922)
bins = 64
theta = (np.arange(bins) + .5) * 2 * np.pi / bins
fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout='constrained')
hist_results = {}
for ax, (label, x, y, color) in zip(axes[0], [('补偿前', .09, -.06, VIOLET), ('补偿后', .002, -.001, TEAL)]):
    mean = 3000 * (1 + x * np.cos(theta) + y * np.sin(theta))
    counts = rng.poisson(mean)
    X, Y = estimate(counts, theta)
    amp = np.hypot(X, Y)
    sigma = np.sqrt(2 / counts.sum())
    ax.errorbar(theta / (2 * np.pi), counts, yerr=np.sqrt(counts), fmt='.', color=color, alpha=.8, label='合成光子计数 ±√N')
    fine = np.linspace(0, 2 * np.pi, 500)
    ax.plot(fine / (2 * np.pi), counts.mean() * (1 + X * np.cos(fine) + Y * np.sin(fine)), color='#26384e', label='一阶傅里叶拟合')
    ax.set(title=f'{label}：m = {amp:.4f}，σ(X,Y) ≈ {sigma:.4f}', xlabel='RF 相位 / 2π', ylabel='每箱计数', ylim=(2510, 3490))
    ax.legend(fontsize=9)
    hist_results[label] = {'N': int(counts.sum()), 'X': float(X), 'Y': float(Y), 'm': float(amp), 'sigma_XY': float(sigma)}
volt = np.linspace(-.5, .7, 19)
true_x, true_y = .15 * (volt - .12), np.full(volt.size, .018)
obs_x = true_x + rng.normal(0, .002, volt.size)
obs_y = true_y + rng.normal(0, .002, volt.size)
slope, intercept = np.polyfit(volt, obs_x, 1)
optimum = -intercept / slope
axes[1, 0].errorbar(volt, obs_x, yerr=.002, fmt='o', ms=4, color=VIOLET, label='带符号分量 X')
axes[1, 0].errorbar(volt, obs_y, yerr=.002, fmt='s', ms=4, color=TEAL, label='正交分量 Y')
axes[1, 0].plot(volt, slope * volt + intercept, color=VIOLET, lw=1)
axes[1, 0].axhline(0, color=GRAY, ls=':')
axes[1, 0].axvline(optimum, color=GRAY, ls='--')
axes[1, 0].set(title=f'另一组示例：扫描补偿电压，拟合 V* = {optimum:.3f} V', xlabel='某个 DC 通道电压 / V', ylabel='归一化关联分量')
axes[1, 0].legend(fontsize=9)
axes[1, 1].plot(volt, np.hypot(true_x, true_y), color=VIOLET, label='m = √(X²+Y²)')
axes[1, 1].axhline(.018, color=TEAL, ls='--', label='预设的正交残余 0.018')
axes[1, 1].axvline(.12, color=GRAY, ls=':')
axes[1, 1].set(title='幅度最小，并不等于两个分量都为零', xlabel='某个 DC 通道电压 / V', ylabel='调制度 m', ylim=(0, .105))
axes[1, 1].legend(fontsize=9)
fig.suptitle('从 RF 相位直方图到补偿电压  |  全部为合成数据', fontsize=15)
save(fig, 'phase-and-compensation')
checks['synthetic_histograms'] = hist_results
checks['synthetic_scan_optimum_V'] = float(optimum)
assert abs(optimum - .12) < .02

# Fig: finite-frequency atomic response; same beta, scan laser detuning.
deltas = np.linspace(-3, -.01, 500)
s, beta = .05, .03
omega = 25 / 19.6
response = np.array([transfer(d, s, omega) for d in deltas])
adiabatic = abs(8 * deltas * omega / (1 + s + 4 * deltas**2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
axes[0].plot(deltas, abs(response) * beta, color=VIOLET, lw=2, label='线性化 Bloch 动态响应')
axes[0].plot(deltas, adiabatic * beta, color=GRAY, ls='--', label='瞬时洛伦兹斜率近似')
axes[0].set(xlabel='激光失谐 Δ / Γ', ylabel='荧光调制度 m', title='Ωrf / Γ = 25 / 19.6，s = 0.05，β = 0.03')
axes[0].legend(fontsize=9)
phase_relative = np.angle(-response) * 180 / np.pi
axes[1].plot(deltas, phase_relative, color=TEAL, lw=2)
axes[1].axhline(0, color=GRAY, ls='--')
axes[1].set(xlabel='激光失谐 Δ / Γ', ylabel='相对准静态响应的相位 / °', title='相位也随失谐变化：需同时标定幅度与相位')
fig.suptitle('RF 频率接近自然线宽时，原子响应不能视为瞬时  |  理想二能级模型', fontsize=14)
save(fig, 'dynamic-response')

# Worked example: assumed parameters, not actual settings of the user's trap.
lam, gamma, om, charge, mass, c = 369.5e-9, 2*np.pi*19.6e6, 2*np.pi*25e6, 1.602176634e-19, 172*1.66053906660e-27, 299792458.
xobs, yobs, dilution = .015, .020, .8
m_sig = np.hypot(xobs, yobs) / dilution
h = abs(transfer(-.5, .05, om/gamma))
b = m_sig / h
r = b * lam / (2*np.pi)
v = om * r
checks['worked_example'] = {'m_signal': float(m_sig), 'H_magnitude': float(h), 'beta': float(b), 'r_peak_nm': float(r*1e9), 'v_peak_m_s': float(v), 'E_rf_peak_V_m': float(mass*om**2*r/charge), 'fractional_doppler_projection': float(-v*v/(4*c*c)), 'sigma_component_N200000_eta08': float(np.sqrt(2/200000)/.8)}
checks['method'] = 'Analytic steady state, adiabatic limit, weak-drive sidebands, independent time-domain ODE, and synthetic scan recovery.'
(HERE / 'validation.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(checks, indent=2, ensure_ascii=False))
