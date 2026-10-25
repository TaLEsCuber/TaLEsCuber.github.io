"""Reproduce the AM/PM tutorial's analytical simulations and numerical checks.

Requires Python 3.10+, numpy, scipy, matplotlib. No measurements or random data.
Run: python figures-and-checks.py [--output DIRECTORY] [--qa-tools DIRECTORY]
The optional QA tools perform panel-alignment, PDF text, and collision audits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import jv, jn_zeros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--qa-tools", type=Path)
args = parser.parse_args()
if args.output:
    IMG, DATA, QA = (args.output / p for p in ("figures", "data", "qa"))
else:
    ROOT = Path(__file__).absolute().parents[3]
    IMG = ROOT / "source/img/amplitude-phase-modulation"
    DATA = Path(__file__).absolute().parent
    QA = ROOT / "tmp/am-pm/qa"
for directory in (IMG, DATA, QA):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8, "axes.unicode_minus": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "#23374d", "axes.labelcolor": "#23374d",
    "savefig.dpi": 300,
})
BLUE, ORANGE, TEAL, GRAY = "#3369a8", "#cb7428", "#008b8b", "#8593a3"
reports = []
contracts = []


def figure(rows, cols, title, footer, height=None):
    fig, axes = plt.subplots(rows, cols, figsize=(12, height or (3.1 * rows + 1.7)), squeeze=False)
    fig.subplots_adjust(left=.09, right=.965, bottom=.17 if rows < 4 else .105,
                        top=.79 if rows == 1 else .88, wspace=.35, hspace=.7)
    fig.text(.09, .97, title, ha="left", va="top", fontsize=18, weight="bold")
    fig.text(.09, .027, footer, ha="left", va="bottom", fontsize=11)
    return fig, axes


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", pad=12)
    ax.grid(axis="y", alpha=.15)


def save(fig, name, claim, roles):
    fig.canvas.draw()
    if args.qa_tools:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        alignment = require_matplotlib_panel_alignment(
            fig, json_out=QA / f"{name}.alignment.json",
            overlay_svg=QA / f"{name}.alignment.svg", strict=True,
            tolerance_pt=1.5, gutter_tolerance_pt=1.5)
        reports.append({"figure": name, "alignment": alignment.get("verdict", alignment.get("status"))})
    fig.savefig(IMG / f"{name}.png", dpi=300)
    fig.savefig(IMG / f"{name}.svg")
    fig.savefig(QA / f"{name}.pdf")
    if args.qa_tools:
        commands = [
            ["audit_pdf_text.py", str(QA / f"{name}.pdf"), "--min-pt", "5", "--json"],
            ["audit_figure_collisions.py", str(QA / f"{name}.pdf"),
             "--json-out", str(QA / f"{name}.collision.json"),
             "--overlay-pdf", str(QA / f"{name}.collision.pdf")],
        ]
        for command in commands:
            bootstrap = ("import runpy,sys; sys.path[:0]=" + repr(sys.path) +
                         "; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
            result = subprocess.run([sys.executable, "-c", bootstrap, str(args.qa_tools / command[0]), *command[1:]],
                                    capture_output=True, text=True, encoding="utf-8")
            (QA / f"{name}.{command[0]}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
            reports.append({"figure": name, "audit": command[0], "exit_code": result.returncode})
    contracts.append({"figure": name, "claim": claim, "panels": roles,
                      "archetype": "quantitative grid", "source": "analytical simulation",
                      "uncertainty": "none; deterministic equations", "backend": "Python",
                      "export": "12-inch web teaching figure; PNG 300 dpi; editable SVG; QA PDF"})
    plt.close(fig)


tau = np.linspace(0, 2, 5000)
theta = 2 * np.pi * tau
message = np.cos(theta)
carrier_phase = 2 * np.pi * 12 * tau

# 1. Controlled comparison: same message, amplitude versus phase encoding.
fig, ax = figure(3, 1, "同一个消息信号，可以改变振幅，也可以改变相位",
                "教学显示取 fc/fm = 12；蓝色为 AM，青色为 PM，橙色虚线为包络或参考载波。", 9.2)
panel(ax[0, 0], "a", "消息信号：m(t) = cos(2πfmt)")
ax[0, 0].plot(tau, message, color=GRAY, lw=2)
ax[0, 0].set_ylabel("归一化消息")
panel(ax[1, 0], "b", "AM：μ = 0.6，峰值随消息变化")
env = 1 + .6 * message
ax[1, 0].plot(tau, env * np.cos(carrier_phase), color=BLUE, lw=1.1)
ax[1, 0].plot(tau, env, "--", color=ORANGE, lw=1.8)
ax[1, 0].plot(tau, -env, "--", color=ORANGE, lw=1.8)
ax[1, 0].set_ylabel("sAM / Ac")
panel(ax[2, 0], "c", "PM：β = 1 rad，包络恒定，零点位置移动")
ax[2, 0].plot(tau, np.cos(carrier_phase), color=GRAY, lw=.8, alpha=.6, ls="--")
ax[2, 0].plot(tau, np.cos(carrier_phase + message), color=TEAL, lw=1.3)
for a in ax[:, 0]:
    a.set_xlim(0, 2)
    a.set_xlabel("时间 t / Tm")
    a.axhline(0, color=GRAY, lw=.5)
ax[2, 0].set_ylabel("sPM / Ac")
ax[1, 0].set_ylim(-1.9, 1.9)
ax[2, 0].set_ylim(-1.25, 1.25)
save(fig, "01-time-domain", "同一消息在 AM 中改变包络，在 PM 中改变载波相位。",
     ["消息参考", "振幅编码", "相位编码和未调制参考"])

# 2. Overmodulation is a sign loss at envelope detection, not compulsory clipping.
fig, ax = figure(3, 1, "过调制：包络检波丢掉了振幅系数的正负号",
                "蓝色：AM 波形；橙色虚线：带符号的 1+μm(t)；黑色点线：检波器读到的绝对值。", 9.4)
for i, mu in enumerate((.3, 1., 1.4)):
    a = ax[i, 0]
    signed = 1 + mu * message
    panel(a, chr(97+i), f"μ = {mu:g}：" + ("正常调制" if mu < 1 else "临界调制" if mu == 1 else "过调制，负值被折回"))
    a.plot(tau, signed * np.cos(carrier_phase), color=BLUE, lw=.8)
    a.plot(tau, signed, color=ORANGE, ls="--", lw=2.1)
    a.plot(tau, np.abs(signed), color="#29313b", ls=":", lw=1.8)
    a.axhline(0, color=GRAY, lw=.6)
    a.set(xlim=(0, 2), ylim=(-2.6, 2.8), xlabel="时间 t / Tm", ylabel="归一化幅度")
save(fig, "02-am-depth", "μ>1 时有符号包络过零，绝对值检波产生折叠失真。",
     ["正常参考", "过零边界", "负包络和折叠的失效机制"])

# 3. Discrete lines and a general bandlimited message are two different cases.
fig, ax = figure(2, 1, "AM 的频谱：相乘，把基带搬到载波两侧",
                "仅画正载频附近；上图为单音的余弦峰值幅度，下图为连续谱的形状示意，二者纵轴不可混用。", 7.4)
panel(ax[0, 0], "a", "单音 AM：载波 1，上下边带各 μ/2 = 0.3")
ax[0, 0].vlines([-1, 0, 1], 0, [.3, 1, .3], color=[BLUE, GRAY, BLUE], lw=4)
ax[0, 0].scatter([-1, 0, 1], [.3, 1, .3], color=[BLUE, GRAY, BLUE], s=40, zorder=3)
ax[0, 0].set(xlim=(-2.2, 2.2), ylim=(0, 1.23), ylabel="余弦峰值 / Ac", xlabel="频率偏移 (f − fc) / fm")
ax[0, 0].set_xticks([-1, 0, 1], ["−1：下边带", "0：载波", "+1：上边带"])
panel(ax[1, 0], "b", "一般基带：最高频率 Bm，对应 AM 带宽 2Bm")
u = np.linspace(-1, 1, 800)
shape = .46 * (1-np.abs(u)) * (1+.2*np.cos(3*np.pi*u))
ax[1, 0].fill_between(u, shape, color=BLUE, alpha=.2)
ax[1, 0].plot(u, shape, color=BLUE, lw=2)
ax[1, 0].vlines(0, 0, 1, color=GRAY, lw=3)
ax[1, 0].set(xlim=(-2.2, 2.2), ylim=(0, 1.23), ylabel="谱形示意", xlabel="频率偏移 (f − fc) / Bm")
ax[1, 0].set_xticks([-1, 0, 1], ["fc − Bm", "fc", "fc + Bm"])
save(fig, "03-am-spectrum", "AM 的两侧频移副本决定双边带的 2Bm 带宽。",
     ["单音三条谱线", "一般带限消息的频谱支持范围"])

# 4. Complex-envelope geometry; fixed plot boxes avoid aspect-driven misalignment.
fig, ax = figure(1, 2, "复平面：AM 沿半径变化，PM 沿圆弧变化",
                "已去除载波旋转。点表示 m(t)=1 的瞬间；虚线为坐标参考。", 5.6)
for a in ax.flat:
    a.axhline(0, color=GRAY, lw=.7, ls="--")
    a.axvline(0, color=GRAY, lw=.7, ls="--")
    a.set(xlim=(-.4, 1.8), ylim=(-1.1, 1.1), xlabel="I / Ac", ylabel="Q / Ac")
    a.set_aspect("equal", adjustable="box")
panel(ax[0, 0], "a", "AM：z/Ac = 1 + μm，μ = 0.6")
ax[0, 0].plot([.4, 1.6], [0, 0], color=BLUE, lw=5)
ax[0, 0].quiver(0, 0, 1.6, 0, angles="xy", scale_units="xy", scale=1, color=BLUE, width=.009)
ax[0, 0].scatter([1.6], [0], color=ORANGE, s=70, zorder=4)
panel(ax[0, 1], "b", "PM：z/Ac = exp[jβm]，β = 1 rad")
angle = np.linspace(-1, 1, 500)
ax[0, 1].plot(np.cos(angle), np.sin(angle), color=TEAL, lw=4)
ax[0, 1].quiver(0, 0, np.cos(1), np.sin(1), angles="xy", scale_units="xy", scale=1, color=TEAL, width=.009)
ax[0, 1].scatter([np.cos(1)], [np.sin(1)], color=ORANGE, s=70, zorder=4)
save(fig, "04-iq-geometry", "AM 与 PM 分别对应复包络的径向变化和角向变化。",
     ["AM 轨迹", "PM 轨迹"])

# 5. Derivative relation prevents confusing phase extrema with frequency extrema.
fig, ax = figure(3, 1, "PM 的瞬时频率由相位的斜率决定",
                "PM：相位 βcos(2πfmt)，频偏 −βfm sin(2πfmt)。消息到达极值时，PM 的频偏恰为零。", 9.2)
signals = [message, message, -np.sin(theta)]
titles = ["消息 m(t)", "附加相位 φ(t) / β = m(t)", "瞬时频偏 [fi(t) − fc] / (βfm)"]
for i, (values, title) in enumerate(zip(signals, titles)):
    panel(ax[i, 0], chr(97+i), title)
    ax[i, 0].plot(tau, values, color=(GRAY, TEAL, ORANGE)[i], lw=2)
    ax[i, 0].set(xlim=(0, 2), ylim=(-1.25, 1.25), xlabel="时间 t / Tm", ylabel="归一化量")
    ax[i, 0].axhline(0, color=GRAY, lw=.5)
    for point in (.25, .75, 1.25, 1.75):
        ax[i, 0].axvline(point, color=GRAY, alpha=.25, lw=.8, ls="--")
save(fig, "05-phase-frequency", "PM 的频偏正比于消息的时间导数。",
     ["消息输入", "相位响应", "导数给出的瞬时频偏"])

# 6. Absolute powers rather than dBc remain well-defined at a carrier null.
fig, ax = figure(4, 1, "PM 的边带：深度越大，高阶谱线越重要",
                "每条谱线的功率以未调制载波功率 Pc 为基准：Pn/Pc = Jn(β)²；上下边带各算一条。", 11.4)
orders = np.arange(-8, 9)
root = float(jn_zeros(0, 1)[0])
for i, beta in enumerate((.2, 1., root, 4.)):
    a = ax[i, 0]
    panel(a, chr(97+i), f"β = {beta:.4g} rad" + ("：载波达到第一个零点" if i == 2 else ""))
    p = jv(orders, beta)**2
    colors = [GRAY if n == 0 else TEAL for n in orders]
    a.bar(orders, p, width=.28, color=colors)
    a.set(xlim=(-8.7, 8.7), ylim=(0, 1.05), xlabel="边带阶数 n；频率为 fc + nfm", ylabel="Pn / Pc")
    a.set_xticks(np.arange(-8, 9, 2))
save(fig, "06-pm-sidebands", "PM 深度增加时功率在载波及高阶边带之间重新分配。",
     ["小深度", "中等深度", "载波零点", "多阶边带"])

# 7. A zero carrier does not imply zero total power.
fig, ax = figure(1, 1, "载波消失时，功率去了哪里？",
                "蓝色：载波；青色：±1 阶边带合计；橙色：其余边带合计。三者之和恒为 1。", 5.3)
b = np.linspace(0, 5, 1000)
pc, p1 = jv(0, b)**2, 2*jv(1, b)**2
ax[0, 0].plot(b, pc, color=BLUE, lw=2.4, label="载波（零阶）")
ax[0, 0].plot(b, p1, color=TEAL, lw=2.4, label="±1 阶边带合计")
ax[0, 0].plot(b, 1-pc-p1, color=ORANGE, lw=2.2, label="其余边带")
ax[0, 0].axvline(root, color=GRAY, lw=1, ls="--")
ax[0, 0].set(xlim=(0, 5), ylim=(0, 1.08), xlabel="相位调制深度 β / rad", ylabel="相对功率")
ax[0, 0].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3)
save(fig, "07-pm-power", "PM 的载波零点是功率重分配，而不是总功率消失。",
     ["各阶功率份额随深度的变化"])

# 8. Exact detectors versus approximations in two known failure regimes.
fig, ax = figure(2, 1, "解调：知道读出的究竟是包络、正交分量，还是相位",
                "黑色虚线为原消息；上图比较 AM 检波方式，下图比较 PM 的线性近似与完整相位读出。", 7.9)
mu, beta = 1.4, 2.
panel(ax[0, 0], "a", "过调制 AM：μ = 1.4，包络检波失真，相干解调保留符号")
ax[0, 0].plot(tau, message, color="#29313b", lw=3.3, ls="--", label="原消息")
ax[0, 0].plot(tau, message, color=BLUE, lw=1.5, label="相干解调")
ax[0, 0].plot(tau, (np.abs(1+mu*message)-1)/mu, color=ORANGE, lw=2, label="包络检波后归一化")
panel(ax[1, 0], "b", "PM：β = 2 rad，只取 Q/β 会失真，atan2(Q,I)/β 恢复消息")
ax[1, 0].plot(tau, message, color="#29313b", lw=3.3, ls="--", label="原消息")
ax[1, 0].plot(tau, message, color=TEAL, lw=1.5, label="完整相位解调")
ax[1, 0].plot(tau, np.sin(beta*message)/beta, color=ORANGE, lw=2, label="Q/β 的线性近似")
for a in ax[:, 0]:
    a.set(xlim=(0, 2), ylim=(-1.25, 1.65), xlabel="时间 t / Tm", ylabel="恢复的消息")
    a.legend(loc="upper center", ncol=3)
save(fig, "08-demodulation", "非线性检波或小信号近似会在明确条件下失真。",
     ["AM 过调制的两种读出", "PM 大深度的两种读出"])

# 9. Optical square-law detection and an interferometric phase-to-power conversion.
fig, ax = figure(2, 1, "光学测量：电场振幅、光强和相位是不同的量",
                "上图：直接光电探测；下图：理想干涉仪在正交点把相位变化转换成输出功率变化。", 7.8)
panel(ax[0, 0], "a", "场 AM 深度 μE = 0.3，与理想纯 PM 的直接光强")
ax[0, 0].plot(tau, (1+.3*message)**2, color=BLUE, lw=2.1, label="场 AM：I/Ic = (1+0.3m)²")
ax[0, 0].plot(tau, np.ones_like(tau), color=TEAL, lw=2.1, label="纯 PM：I/Ic = 1")
ax[0, 0].set(xlim=(0, 2), ylim=(.25, 2.15), xlabel="时间 t / Tm", ylabel="直接探测的 I / Ic")
ax[0, 0].legend(loc="upper center", ncol=2)
panel(ax[1, 0], "b", "正交偏置的干涉仪：φb = π/2，差分相位调制深度 0.6 rad")
ax[1, 0].plot(tau, .5*(1+np.cos(np.pi/2+.6*message)), color=TEAL, lw=2.2, label="完整传递函数")
ax[1, 0].plot(tau, .5-.3*message, color=ORANGE, lw=1.6, ls="--", label="小信号近似")
ax[1, 0].set(xlim=(0, 2), ylim=(0, 1.12), xlabel="时间 t / Tm", ylabel="输出功率 Pout / Pin")
ax[1, 0].legend(loc="upper center", ncol=2)
for a in ax[:, 0]:
    a.grid(False)
save(fig, "09-optical-detection", "平方律探测区分场 AM 与光强调制，干涉读出才能将纯 PM 转成光强变化。",
     ["直接探测", "相位到功率的干涉转换"])

# Numerical checks are independent of the plotted line coordinates where practical.
N = 65536
th = 2*np.pi*np.arange(N)/N
mu = .6
v = (1+mu*np.cos(th))*np.cos(100*th)
expanded = np.cos(100*th) + mu/2*np.cos(101*th) + mu/2*np.cos(99*th)
am_error = float(np.max(np.abs(v-expanded)))
assert am_error < 1e-12
am_fft = np.fft.fft(1+mu*np.cos(th))/N
assert np.max(np.abs(am_fft[[0, 1, -1]]-np.array([1., .3, .3]))) < 1e-12
pm_coeff_errors, pm_power_errors = {}, {}
for beta in (0., .05, .2, 1., root, 4., 8.):
    z = np.exp(1j*beta*np.cos(th))
    fourier = np.fft.fft(z)/N
    n = np.arange(-50, 51)
    coeff = (1j)**n * jv(n, beta)
    residual = float(np.max(np.abs(fourier[n % N]-coeff)))
    power_error = float(abs(np.sum(jv(n, beta)**2)-1))
    assert residual < 1e-12 and power_error < 1e-12
    pm_coeff_errors[str(beta)] = residual
    pm_power_errors[str(beta)] = power_error

# Passband generation -> mixers -> ideal FFT low-pass -> message recovery.
samples = 40000
fs, fc, fm = 20e6, 1e6, 10e3
t = np.arange(samples)/fs
m = np.cos(2*np.pi*fm*t)
mix = 2*np.pi*fc*t
freq = np.fft.fftfreq(samples, 1/fs)
mask = np.abs(freq) < 200e3
def lowpass(x):
    return np.fft.ifft(np.fft.fft(x)*mask).real
signal_am = (1+1.4*m)*np.cos(mix)
signal_pm = np.cos(mix+2*m)
am_read = (lowpass(2*signal_am*np.cos(mix))-1)/1.4
I = lowpass(2*signal_pm*np.cos(mix))
Q = lowpass(-2*signal_pm*np.sin(mix))
pm_read = np.unwrap(np.angle(I+1j*Q))/2
am_recovery_error = float(np.max(np.abs(am_read-m)))
pm_recovery_error = float(np.max(np.abs(pm_read-m)))
assert am_recovery_error < 1e-10 and pm_recovery_error < 1e-10
envelope_rmse = float(np.sqrt(np.mean(((np.abs(1+1.4*m)-1)/1.4-m)**2)))
linear_pm_rmse = float(np.sqrt(np.mean((np.sin(2*m)/2-m)**2)))
assert abs(jv(0, root)) < 1e-14
Pc = 1**2/(2*50)
first_sideband_ratio = abs(float(jv(1, 1)/jv(0, 1)))
if not 0 < first_sideband_ratio:
    raise ValueError("The sideband-to-carrier ratio must be strictly positive before taking log10.")
example = {
    "carrier_frequency_Hz": fc, "message_frequency_Hz": fm, "Ac_peak_V": 1, "R_ohm": 50,
    "unmodulated_carrier_W": Pc,
    "AM_mu": .6, "AM_total_W": Pc*(1+.6**2/2), "AM_each_sideband_W": Pc*.6**2/4,
    "AM_efficiency": (.6**2/2)/(1+.6**2/2),
    "PM_beta_rad": 1, "PM_carrier_W": float(Pc*jv(0, 1)**2),
    "PM_each_first_sideband_W": float(Pc*jv(1, 1)**2),
    "PM_first_sideband_dBc": float(20*np.log10(first_sideband_ratio)),
    "PM_peak_frequency_deviation_Hz": fm, "PM_Carson_estimate_Hz": 4*fm,
}
validation = {
    "data_kind": "deterministic analytical simulations; not experimental observations",
    "am_expansion_max_error": am_error,
    "pm_fourier_coefficient_max_errors": pm_coeff_errors,
    "pm_parseval_power_errors": pm_power_errors,
    "carrier_first_null_beta_rad": root,
    "passband_am_recovery_max_error": am_recovery_error,
    "passband_pm_recovery_max_error": pm_recovery_error,
    "overmodulated_envelope_message_RMSE": envelope_rmse,
    "large_beta_linear_Q_message_RMSE": linear_pm_rmse,
    "example": example,
    "figure_count": len(contracts), "figure_audits": reports,
}
(DATA / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
(DATA / "figure-contracts.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
np.savetxt(DATA / "pm-beta1-sidebands.csv", np.column_stack((orders, jv(orders, 1), jv(orders, 1)**2)),
           delimiter=",", header="n,Jn_beta1,power_fraction_of_Pc", comments="")
if any(row.get("exit_code", 0) != 0 for row in reports):
    raise RuntimeError("Figure audit failed; inspect the generated QA reports before delivery.")
print(json.dumps({"figures": len(contracts), "am_error": am_error,
                  "pm_fourier_error": max(pm_coeff_errors.values()),
                  "am_recovery_error": am_recovery_error, "pm_recovery_error": pm_recovery_error,
                  "example": example}, ensure_ascii=False, indent=2))
