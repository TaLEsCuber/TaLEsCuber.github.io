"""Reproduce the PSF tutorial figures and independent numerical checks.

Requires Python 3.10+, numpy, scipy, matplotlib. All images are simulations.
Run this file in the blog checkout, or use --output OUTPUT_DIRECTORY anywhere.
Optional --qa-tools DIRECTORY enables external figure layout/PDF audits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--qa-tools", type=Path)
args = parser.parse_args()
if args.output:
    IMG = args.output / "figures"
    DATA = args.output / "data"
    QA = args.output / "qa"
else:
    ROOT = Path(__file__).resolve().parents[3]
    IMG = ROOT / "source/img/point-spread-function"
    DATA = Path(__file__).resolve().parent
    QA = ROOT / "tmp/psf-qa"
for directory in (IMG, DATA, QA):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "axes.linewidth": .8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "#20334b", "axes.labelcolor": "#20334b",
    "legend.frameon": False, "legend.fontsize": 10,
    "savefig.dpi": 300,
})
BLUE, TEAL, ORANGE, GRAY = "#3366a8", "#008b8b", "#d88730", "#8996a5"
COLORS = (BLUE, TEAL, ORANGE)
alignment_reports = []

def airy_amplitude(r):
    v = 2 * np.pi * np.asarray(r, dtype=float)
    out = np.ones_like(v)
    np.divide(2 * j1(v), v, out=out, where=np.abs(v) > 1e-14)
    return out

def airy(r):
    return airy_amplitude(r) ** 2

def mtf(nu):
    n = np.clip(np.asarray(nu, dtype=float), 0, 1)
    return 2 / np.pi * (np.arccos(n) - n * np.sqrt(1 - n * n))

def figure(rows, cols, title, footer, height=None):
    fig, axes = plt.subplots(rows, cols, figsize=(12, height or (4.9 * rows)), squeeze=False)
    fig.subplots_adjust(left=.075, right=.965, bottom=.16 if rows == 1 else .17,
                        top=.80 if rows == 1 else .88, wspace=.42, hspace=.55)
    fig.text(.075, .97, title, ha="left", va="top", fontsize=18, weight="bold")
    fig.text(.075, .035, footer, ha="left", va="bottom", fontsize=11)
    return fig, axes

def save(fig, name):
    fig.canvas.draw()
    if args.qa_tools:
        spec = importlib.util.spec_from_file_location("alignment", args.qa_tools / "audit_panel_alignment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.require_matplotlib_panel_alignment(
            fig, json_out=QA / f"{name}.alignment.json", strict=True,
            tolerance_pt=1.5, gutter_tolerance_pt=1.5)
        alignment_reports.append({"figure": name, "verdict": result.get("verdict")})
    fig.savefig(IMG / f"{name}.png", dpi=300)
    fig.savefig(IMG / f"{name}.svg")
    fig.savefig(IMG / f"{name}.pdf")
    plt.close(fig)
    print("Rendered", name, flush=True)

def heat(ax, im, extent, title, vmax=1, vmin=0):
    ax.imshow(im, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("x / (λ0/NA)")
    ax.set_ylabel("y / (λ0/NA)")

def line_axis(ax, title, xlabel="x / (λ0/NA)", ylabel="相对强度"):
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e7ecf0", linewidth=.6)

# 01: exact weighted sum of shifted Airy intensity kernels.
grid = np.linspace(-2, 2, 501)
X, Y = np.meshgrid(grid, grid)
points = [(-.8, .55, 1.), (.7, .45, .7), (-.25, -.65, .85), (.25, -.65, .85)]
image = sum(weight * airy(np.hypot(X - x, Y - y)) for x, y, weight in points)
fig, aa = figure(1, 3, "01  一个物点，贡献一个平移的 PSF",
    "示意为互不相干的点源；图 c 是四个按亮度加权的 Airy 强度之和。", height=5.4)
ax = aa[0, 0]
ax.set_facecolor("#10101d")
ax.scatter([p[0] for p in points], [p[1] for p in points],
           s=[60 * p[2] for p in points], c="#fff0be")
ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="x / (λ0/NA)", ylabel="y / (λ0/NA)")
ax.set_aspect("equal")
ax.set_title("a  物体：四个理想点源", loc="left", pad=12)
heat(aa[0, 1], airy(np.hypot(X, Y)), (-2, 2, -2, 2), "b  单个点源的响应")
heat(aa[0, 2], image, (-2, 2, -2, 2), "c  平移、加权、叠加", vmax=image.max())
save(fig, "01-point-to-image")

# 02: analytical Airy intensity, with explicit peak versus energy normalization.
rr = np.linspace(0, 2.5, 1201)
ee = 1 - j0(2 * np.pi * rr) ** 2 - j1(2 * np.pi * rr) ** 2
r_zero = float(jn_zeros(1, 1)[0] / (2 * np.pi))
r_half = brentq(lambda r: float(airy(r)) - .5, .1, .4)
fwhm = 2 * r_half
fig, aa = figure(2, 2, "03  Airy 斑：宽度、旁瓣和能量分布",
    "上排同一 PSF：左为线性显示，右为 log10 显示（−4 至 0）。下排采用解析公式。", height=11)
heat(aa[0, 0], airy(np.hypot(X, Y)), (-2, 2, -2, 2), "a  线性强度：0 至 1")
heat(aa[0, 1], np.log10(np.maximum(airy(np.hypot(X, Y)), 1e-4)),
     (-2, 2, -2, 2), "b  对数强度：旁瓣更明显", vmin=-4)
ax = aa[1, 0]
ax.set_box_aspect(1)
ax.plot(rr, airy(rr), c=BLUE, lw=2)
ax.axhline(.5, c=GRAY, ls="--", lw=.9)
ax.axvline(r_zero, c=ORANGE, ls="--", lw=1.1)
ax.set(xlim=(0, 1.7), ylim=(0, 1.06))
line_axis(ax, "c  径向剖面", "r / (λ0/NA)", "h(r) / h(0)")
ax.grid(False)
ax.text(.96, .86, "FWHM = 0.514 λ0/NA\n首零半径 = 0.610 λ0/NA",
        ha="right", va="top", transform=ax.transAxes, fontsize=11)
ax = aa[1, 1]
ax.set_box_aspect(1)
ax.plot(rr, ee, c=TEAL, lw=2)
ax.axhline(.8378, c=GRAY, ls="--", lw=.9)
ax.axvline(r_zero, c=ORANGE, ls="--", lw=1.1)
ax.set(xlim=(0, 2.5), ylim=(0, 1.05))
line_axis(ax, "d  包围能量", "R / (λ0/NA)", "E(R)")
ax.grid(False)
ax.text(.96, .25, "首暗环以内约含 83.8% 能量", ha="right", transform=ax.transAxes, fontsize=11)
save(fig, "02-airy-anatomy")
np.savetxt(DATA / "airy-profile.csv", np.c_[rr, airy(rr), ee], delimiter=",",
           header="r_in_lambda_over_NA,peak_normalized_intensity,encircled_energy", comments="")

# 03: two equally bright mutually incoherent points; common scale for all panels.
xp = np.linspace(-1.5, 1.5, 2001)
xg = np.linspace(-1.5, 1.5, 401)
yg = np.linspace(-.6, .6, 161)
XX, YY = np.meshgrid(xg, yg)
distances = [.40, r_zero, .90]
fig, aa = figure(2, 3, "04  两个点能否分开，要看间距和判据",
    "等亮非相干点源；所有像与曲线共用单点峰值单位。虚线标记真实点源位置。", height=7.6)
dips = []
for k, d in enumerate(distances):
    pair = airy(np.hypot(XX - d / 2, YY)) + airy(np.hypot(XX + d / 2, YY))
    heat(aa[0, k], pair, (-1.5, 1.5, -.6, .6), f"{chr(97+k)}  间距 {d:.3f} λ0/NA", vmax=1.6)
    left, right = airy(xp - d / 2), airy(xp + d / 2)
    ax = aa[1, k]
    ax.plot(xp, left, color=GRAY, ls="--", lw=1)
    ax.plot(xp, right, color=GRAY, ls=":", lw=1)
    ax.plot(xp, left + right, color=COLORS[k], lw=2)
    for pos in (-d / 2, d / 2):
        ax.axvline(pos, color=GRAY, lw=.8, ls="--")
    ax.set(xlim=(-1.5, 1.5), ylim=(0, 1.6))
    line_axis(ax, ["d  中央仍是单峰", "e  Rayleigh 间距", "f  双峰分离较明显"][k])
    dips.append(float(1 - 2 * airy(d / 2) / np.max(left + right)))
save(fig, "03-two-point-resolution")

# 04: phase-sensitive addition of coherent amplitudes.
d = r_zero
a1, a2 = airy_amplitude(xp - d / 2), airy_amplitude(xp + d / 2)
cases = [a1*a1 + a2*a2, (a1+a2)**2, (a1-a2)**2]
fig, aa = figure(1, 3, "02  同一对物点，相干性改变像强度",
    "单点振幅相同、间距为 Rayleigh 间距；纵轴共用相同单位，未分别归一化。", height=5.2)
for k, vals in enumerate(cases):
    ax = aa[0, k]
    ax.plot(xp, vals, color=COLORS[k], lw=2)
    ax.set(xlim=(-1.5, 1.5), ylim=(0, 1.7))
    line_axis(ax, ["a  非相干：强度相加", "b  相干：同相", "c  相干：反相"][k])
save(fig, "04-coherence")

# 05: incoherent circular pupil MTF and sinusoidal test objects.
nu = np.linspace(0, 1.2, 1001)
fig, aa = figure(1, 2, "05  MTF：每种空间频率保留多少对比度",
    "圆瞳、无像差、非相干成像。右图每条正弦单独加了纵向偏置，以便阅读。", height=5.8)
ax = aa[0, 0]
ax.plot(nu, mtf(nu), color=BLUE, lw=2)
ax.plot([.5], [mtf(.5)], "o", color=ORANGE)
ax.text(.52, .46, "MTF(0.5) ≈ 0.391", fontsize=11)
ax.set(xlim=(0, 1.2), ylim=(0, 1.03))
line_axis(ax, "a  理想圆瞳 MTF", "归一化频率 ν = f/fc", "MTF")
ax = aa[0, 1]
xs = np.linspace(0, 6, 1500)
for i, q in enumerate((.2, .5, .9)):
    offset = (2 - i) * 2.7
    wave = np.cos(2 * np.pi * q * xs)
    ax.plot(xs, offset + wave, color=GRAY, lw=1, ls="--")
    ax.plot(xs, offset + float(mtf(q)) * wave, color=COLORS[i], lw=2)
    ax.text(6.4, offset, f"ν={q:.1f}\nMTF={float(mtf(q)):.3f}",
            color=COLORS[i], fontsize=10, va="center")
ax.set(xlim=(0, 9.7), ylim=(-1.4, 7))
line_axis(ax, "b  正弦物体的输出", "x · fc", "偏置后的调制度")
ax.grid(False)
save(fig, "05-mtf")
np.savetxt(DATA / "circular-mtf.csv", np.c_[nu, mtf(nu)], delimiter=",",
           header="frequency_over_incoherent_cutoff,MTF", comments="")

# 06: sampled generalized pupil, FFT intensity; equal throughput and common scale.
N = 1024
L = 8.0
p = (np.arange(N) - N//2) * L/N
PX, PY = np.meshgrid(p, p)
R = np.hypot(PX, PY)
mask = R <= 1
zernike = [np.zeros_like(R), np.sqrt(3)*(2*R**2-1),
           np.sqrt(6)*(PX**2-PY**2), np.sqrt(8)*(3*R**2-2)*PX]
waves = []
psfs = []
phase_rms = .12
for k, zern in enumerate(zernike):
    w = np.zeros_like(R)
    if k:
        zern = zern - np.mean(zern[mask])
        w[mask] = phase_rms * zern[mask] / np.std(zern[mask])
    pupil = mask * np.exp(2j*np.pi*w)
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf = np.abs(field)**2
    psf /= psf.sum()
    waves.append(w)
    psfs.append(psf)
peak0 = psfs[0].max()
strehl = [float(h.max()/peak0) for h in psfs]
q = np.fft.fftshift(np.fft.fftfreq(N, d=L/N))
crop = np.abs(q) <= 2
pcrop = np.abs(p) <= 1.15
fig, aa = figure(2, 4, "06  瞳面相位如何改变 PSF",
    "上排：波前误差 W/λ0（−0.4 至 0.4）；下排：相对理想峰值的 log10 强度（−4 至 0）。\n"
    "三种像差均为 0.12 λ0 RMS；全网格能量相等。展示范围只截取中心，不重新归一化。", height=8.2)
for k, title in enumerate(("理想", "离焦", "像散", "彗差")):
    ax = aa[0, k]
    w = waves[k].copy()
    w[~mask] = np.nan
    ax.imshow(w[np.ix_(pcrop, pcrop)], extent=(-1.15, 1.15, -1.15, 1.15),
              origin="lower", cmap="coolwarm", vmin=-.4, vmax=.4, aspect="equal")
    ax.set_title(f"{chr(97+k)}  {title}波前", loc="left", pad=12)
    ax.set(xlabel="瞳面 ρx", ylabel="瞳面 ρy")
    heat(aa[1, k], np.log10(np.maximum(psfs[k][np.ix_(crop, crop)]/peak0, 1e-4)),
         (-2.0625, 2.0625, -2.0625, 2.0625),
         f"{chr(101+k)}  S={strehl[k]:.3f}", vmin=-4)
save(fig, "06-aberrations")

# 07: paraxial scalar defocus; axial u=z NA^2/(n lambda0).
rho = np.linspace(0, 1, 1001)
zx = np.linspace(-3, 3, 301)
xx = np.linspace(-1.5, 1.5, 301)
radial_kernel = 2 * j0(2*np.pi*xx[:, None]*rho[None, :])*rho
intensity_xz = []
for u in zx:
    amp = simpson(radial_kernel*np.exp(-1j*np.pi*u*rho**2), x=rho, axis=1)
    intensity_xz.append(np.abs(amp)**2)
intensity_xz = np.asarray(intensity_xz)
fig, aa = figure(1, 2, "07  三维 PSF：横向和轴向尺度不同",
    "标量近轴模型；离焦瞳相位为 −πuρ²。热图 log10 强度范围为 −3 至 0，以焦点峰值为 1。", height=6.0)
ax = aa[0, 0]
ax.imshow(np.log10(np.maximum(intensity_xz, 1e-3)), origin="lower", aspect="auto",
          extent=(-1.5, 1.5, -3, 3), cmap="magma", vmin=-3, vmax=0)
ax.set_title("a  穿过光轴的 xz 截面", loc="left", pad=12)
ax.set(xlabel="x / (λ0/NA)", ylabel="z / (nλ0/NA²)")
ax = aa[0, 1]
ax.plot(zx, np.sinc(zx/2)**2, color=TEAL, lw=2)
ax.plot(zx[::10], intensity_xz[::10, len(xx)//2], "o", ms=3, color=ORANGE, label="瞳面积分")
ax.axhline(.5, c=GRAY, ls="--", lw=.8)
ax.set(xlim=(-3, 3), ylim=(0, 1.06))
line_axis(ax, "b  轴上强度", "z / (nλ0/NA²)", "h(0,0,z) / h(0,0,0)")
ax.grid(False)
ax.legend(loc="upper right")
save(fig, "07-three-dimensional-psf")

# 08: pixel averaging of a PSF line, plus presampling system MTF.
fig, aa = figure(1, 2, "08  像素先积分，再对连续图像采样",
    "左图为一维像素积分示意（沿 y 的探测宽度取极窄极限）；右图是方形像素沿 x 方向的 MTF。", height=5.8)
ax = aa[0, 0]
ax.plot(xp, airy(xp), color=GRAY, lw=1.4, label="连续光学剖面")
for width, color in ((.2, BLUE), (.45, ORANGE)):
    centers = np.arange(-1.35, 1.36, width)
    vals = np.array([quad(lambda x: float(airy(x)), c-width/2, c+width/2)[0]/width
                     for c in centers])
    ax.plot(centers, vals, "o-", color=color, ms=4, lw=1.1,
            label=f"像素宽 {width:.2f} λ0/NA")
ax.set(xlim=(-1.5, 1.5), ylim=(0, 1.08))
line_axis(ax, "a  积分后的样本也依赖亚像素位置")
ax.legend(loc="upper right", fontsize=9)
ax = aa[0, 1]
ax.plot(nu, mtf(nu), color=GRAY, lw=1.4, label="光学 MTF")
for width, color in ((.2, BLUE), (.45, ORANGE)):
    ax.plot(nu, mtf(nu)*np.abs(np.sinc(2*width*nu)), color=color, lw=2,
            label=f"含像素积分：p={width:.2f}")
    ax.axvline(1/(4*width), color=color, lw=1, ls="--", ymax=.32)
ax.set(xlim=(0, 1.4), ylim=(0, 1.05))
line_axis(ax, "b  像素 MTF 与 Nyquist 频率", "ν = f/fc", "MTF")
ax.grid(False)
ax.legend(loc="upper right", fontsize=9)
save(fig, "08-pixel-sampling")

# 09: Poisson image formation and RL with an explicitly matched adjoint.
# Linear convolution with zero extension; H^T 1 corrects finite detector edges.
rng = np.random.default_rng(20261024)
NN = 192
coords = (np.arange(NN) - NN//2)*.08
DX, DY = np.meshgrid(coords, coords)
truth = np.zeros((NN, NN))
truth[((DX+2.5)**2 + DY**2 < 1.4**2) & ((DX+2.5)**2 + DY**2 > 1.05**2)] = .7
truth[(np.abs(DX-1.6)<.10) & (np.abs(DY)<1.7)] = 1
truth[(np.abs(DX-2.2)<.10) & (np.abs(DY)<1.7)] = 1
truth[(np.abs(DY+2.7)<.10) & (np.abs(DX)<3.6)] = .7
for x, y in [(-1.5, 2.7), (-.8, 2.7), (1.7, 2.7), (2.15, 2.7)]:
    truth[np.argmin(abs(coords-y)), np.argmin(abs(coords-x))] = 12
hc = np.arange(-64, 65)*.08
HX, HY = np.meshgrid(hc, hc)
kernel = airy(np.hypot(HX, HY))
kernel /= kernel.sum()  # finite numerical PSF support, documented in article
def forward(a):
    return np.maximum(fftconvolve(a, kernel, mode="same"), 0)
def adjoint(a):
    return np.maximum(fftconvolve(a, kernel[::-1, ::-1], mode="same"), 0)
blurred = forward(truth)
scale, background = 80., .7
observed = rng.poisson(np.maximum(scale*blurred + background, 0)).astype(float)
data = observed/scale
b = background/scale
estimate = np.full_like(truth, max(float(data.mean()-b), 1e-6))
sensitivity = adjoint(np.ones_like(truth))
recons = {}
rl_errors = {}
for it in range(1, 201):
    estimate *= adjoint(data/np.maximum(forward(estimate)+b, 1e-12))/np.maximum(sensitivity, 1e-12)
    if it in (10, 40, 200):
        recons[it] = estimate.copy()
        rl_errors[str(it)] = float(np.sqrt(np.mean((estimate-truth)**2)))
fig, aa = figure(2, 3, "09  反卷积：改善结构，也可能放大噪声",
    "合成数据；统一线性灰度范围 0 至 1.2（点源真值峰值 12 被截亮）。像素间距 0.08 λ0/NA。\n"
    "Poisson 均值为 80×模糊真值 + 0.7；随机种子 20261024；零延拓边界，使用匹配的伴随与边缘灵敏度。", height=9.4)
views = [(truth, "a  合成真值"), (blurred, "b  无噪声的模糊期望"), (data-b, "c  带噪观测（减背景）"),
         (recons[10], "d  RL：10 次迭代"), (recons[40], "e  RL：40 次迭代"), (recons[200], "f  RL：200 次迭代")]
for ax, (im, title) in zip(aa.flat, views):
    heat(ax, im, (coords[0]-.04, coords[-1]+.04, coords[0]-.04, coords[-1]+.04),
         title, vmax=1.2)
    ax.set(xlim=(-5, 5), ylim=(-4, 4))
save(fig, "09-deconvolution")

# Independent checks: roots, integration, real-space quadrature, Fourier-domain
# pupil autocorrelation, matched forward/adjoint, and a physical worked example.
energy_numeric = quad(lambda r: 2*np.pi*r*np.pi*float(airy(r)), 0, r_zero,
                      epsabs=1e-11)[0]
energy_exact = float(1-j0(2*np.pi*r_zero)**2-j1(2*np.pi*r_zero)**2)
norm_v = quad(lambda v: 2*j1(v)**2/v, 0, 10000, limit=20000, epsabs=1e-8)[0]
pupil_errors = []
for r in (.0, .1, .3, .6, 1.2):
    a_num = quad(lambda rho: 2*rho*j0(2*np.pi*r*rho), 0, 1, epsabs=1e-12)[0]
    pupil_errors.append(abs(a_num-float(airy_amplitude(r))))
# MTF numerical area overlap of two unit circular pupils (quadrature over x).
overlap_errors = []
for freq in (.0, .2, .5, .9, 1.0):
    shift = 2*freq
    def thickness(x):
        return 2*min(math.sqrt(max(0, 1-(x-shift/2)**2)),
                     math.sqrt(max(0, 1-(x+shift/2)**2)))
    overlap = quad(thickness, -1+shift/2, 1-shift/2, points=[0], epsabs=1e-10)[0]/np.pi if freq<1 else 0
    overlap_errors.append(abs(overlap-float(mtf(freq))))
test_a = rng.random(truth.shape)
test_b = rng.random(truth.shape)
adjoint_rel = abs(np.sum(forward(test_a)*test_b)-np.sum(test_a*adjoint(test_b)))/np.sum(forward(test_a)*test_b)
axial_error = float(np.max(abs(intensity_xz[:, len(xx)//2] - np.sinc(zx/2)**2)))
lambda_um, NA, M, pixel_um = .550, .65, 40., 6.5
worked = {
    "lambda0_um": lambda_um, "NA": NA, "magnification": M,
    "fwhm_um": fwhm*lambda_um/NA,
    "rayleigh_separation_um": r_zero*lambda_um/NA,
    "incoherent_cutoff_cycles_per_um": 2*NA/lambda_um,
    "object_pixel_um": pixel_um/M,
    "nyquist_pixel_limit_um": lambda_um/(4*NA),
    "nyquist_cycles_per_um": M/(2*pixel_um),
    "pixel_MTF_at_optical_cutoff": float(np.sinc(pixel_um/M*2*NA/lambda_um)),
}
validation = {
    "model": "scalar, uniform circular pupil; mutually incoherent intensity unless stated",
    "figures": 9, "first_zero_lambda_over_NA": r_zero,
    "fwhm_lambda_over_NA": fwhm,
    "encircled_energy_first_zero": energy_exact,
    "energy_integration_abs_error": abs(energy_numeric-energy_exact),
    "normalization_integral_to_v_10000": norm_v,
    "finite_tail_expected_order": "about 6.4e-5 beyond v=10000",
    "pupil_integral_max_abs_error": max(pupil_errors),
    "pupil_overlap_MTF_max_abs_error": max(overlap_errors),
    "MTF_half_cutoff": float(mtf(.5)),
    "Rayleigh_central_dip_fraction": dips[1],
    "FFT_pupil_energy_sums": [float(h.sum()) for h in psfs],
    "aberration_RMS_waves": phase_rms,
    "Strehl_from_sampled_peak": dict(zip(("ideal", "defocus", "astigmatism", "coma"), strehl)),
    "axial_integral_max_abs_error": axial_error,
    "RL_kernel_sum": float(kernel.sum()), "RL_boundary": "zero extension, sensitivity-corrected adjoint",
    "RL_adjoint_relative_error": adjoint_rel, "RL_iterations_RMSE": rl_errors,
    "RL_observed_background_subtracted_RMSE": float(np.sqrt(np.mean((data-b-truth)**2))),
    "random_seed": 20261024, "worked_example": worked,
    "alignment": alignment_reports,
}
assert abs(r_zero-.6098349456) < 1e-9
assert abs(fwhm-.5144969850) < 1e-8
assert abs(energy_exact-.8377848692) < 1e-9
assert max(pupil_errors) < 1e-10 and max(overlap_errors) < 1e-9
assert axial_error < 1e-8 and adjoint_rel < 1e-12
assert all(abs(h.sum()-1)<1e-12 for h in psfs)
assert .9998 < norm_v < 1
(DATA / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(validation, ensure_ascii=True, indent=2), flush=True)
