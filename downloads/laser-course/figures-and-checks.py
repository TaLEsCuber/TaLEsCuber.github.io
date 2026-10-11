"""Reproduce the laser-course figures and independent numerical checks.

Python 3 + numpy + scipy + matplotlib. Run from anywhere:
    python figures-and-checks.py --output-dir PATH
The default output is this script's directory, with a figures/ subdirectory.
The teaching figures are computed, not measured; SI units unless specified.
"""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import eval_hermite, eval_genlaguerre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--figures-dir", type=Path)
args = parser.parse_args()
out = args.output_dir.resolve()
figdir = (args.figures_dir or out / "figures").resolve()
out.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)
fontnames = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if f in fontnames), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "font.size": 12, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
blue, red, green, gold = "#2563a6", "#c54b4b", "#1e846e", "#b68124"
checks = []
rng = np.random.default_rng(20261004)
lam = 632.8e-9
c = 299792458.

def check(name, actual, expected, rtol=1e-9, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    err = float(np.max(np.abs(actual - expected)))
    ok = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    checks.append({"name": name, "passed": ok, "max_absolute_error": err})
    if not ok:
        raise AssertionError(f"{name}: {actual} != {expected}; error={err}")

def T(d):
    return np.array([[1., d], [0., 1.]])

def lens(f):
    return np.array([[1., 0.], [-1/f, 1.]])

def mirror(r):
    return np.array([[1., 0.], [-2/r, 1.]])

def prop(q, m):
    a, b, cc, d = m.ravel()
    return (a*q + b)/(cc*q + d)

def waist(zr, f, ell):
    den = (ell-f)**2 + zr*zr
    return f + f*f*(ell-f)/den, f*f*zr/den

def save(fig, name):
    fig.savefig(figdir / name, dpi=170, bbox_inches="tight")
    plt.close(fig)

# Independent quadratures: circular power, Gaussian pulse and complex overlap.
for a in [0.1, 0.5, 1., 1.5, 2., 4.]:
    integral = quad(lambda r: 4*r*np.exp(-2*r*r), 0, a)[0]
    check(f"aperture_{a}", integral, 1-np.exp(-2*a*a))
pulse_integral = quad(lambda t: np.exp(-4*np.log(2)*t*t), -10, 10)[0]
check("gaussian_pulse_energy", pulse_integral, np.sqrt(np.pi/(4*np.log(2))))
for j in range(20):
    ratio = rng.uniform(.3, 3)
    mismatch = rng.uniform(-4, 4)
    # Use dimensionless r/w2 and mismatch = pi*w1*w2/lambda * Delta(1/R).
    a = 1 + 1/ratio**2 + 1j*mismatch/ratio
    integ = quad(lambda x: (x*np.exp(-a*x*x)).real, 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
    integ += 1j*quad(lambda x: (x*np.exp(-a*x*x)).imag, 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
    numerical_eta = abs(4*integ/ratio)**2
    check(f"mode_overlap_{j}", numerical_eta, 4/((ratio+1/ratio)**2+mismatch**2))

# Random lens tests: independently minimize propagated width using second moments.
for j in range(60):
    zr = rng.uniform(.01, .8)
    f = rng.uniform(.02, .5)
    ell = rng.uniform(-.3, 1.)
    lp, zrp = waist(zr, f, ell)
    sig0 = np.diag([zr, 1/zr])  # common positive scale removed
    m0 = lens(f) @ T(ell)
    sig = m0 @ sig0 @ m0.T
    width_sq = lambda s: sig[0, 0] + 2*s*sig[0, 1] + s*s*sig[1, 1]
    optimum = minimize_scalar(width_sq, bracket=(lp-.01, lp, lp+.01),
                              method="brent", options={"xtol": 1e-12})
    check(f"waist_location_{j}", optimum.x, lp, atol=3e-8)
    check(f"waist_width_{j}", width_sq(lp), zrp, atol=2e-11)
    check(f"q_waist_{j}", prop(1j*zr, T(lp) @ m0), 1j*zrp)
    check(f"back_focal_width_{j}", width_sq(f), f*f/zr)
    f2, d = rng.uniform(.03, .8), rng.uniform(.01, .5)
    m = lens(f) @ T(d) @ lens(f2) @ T(d)
    manual = [[1-d/f2, d*(2-d/f2)],
              [-1/f-1/f2+d/(f*f2), 1-2*d/f-d/f2+d*d/(f*f2)]]
    check(f"period_matrix_{j}", m, manual)
    check(f"period_determinant_{j}", np.linalg.det(m), 1.)
    v = rng.normal(size=2)
    check(f"cayley_hamilton_{j}", (m@m @ v)-np.trace(m)*(m@v)+v, [0, 0], atol=2e-10)
    sig2 = m @ sig0 @ m.T
    check(f"second_moment_invariant_{j}", np.linalg.det(sig2), 1., atol=2e-8)

# GRIN differential equation, weak- and zero-gradient limits, slab interfaces.
gamma, length = 3., .7
G = np.array([[np.cos(gamma*length), np.sin(gamma*length)/gamma],
              [-gamma*np.sin(gamma*length), np.cos(gamma*length)]])
for j in range(10):
    v = rng.normal(size=2)*.01
    result = solve_ivp(lambda z, y: [y[1], -gamma**2*y[0]], [0, length], v,
                       rtol=2e-12, atol=1e-14)
    check(f"grin_ode_{j}", result.y[:, -1], G@v, atol=1e-12)
check("grin_zero_limit", [[np.cos(1e-8*length), np.sin(1e-8*length)/1e-8],
                         [-1e-8*np.sin(1e-8*length), np.cos(1e-8*length)]], T(length))
n, thick = 1.5, .012
slab = np.diag([1., n]) @ T(thick) @ np.diag([1., 1/n])
check("slab_matrix", slab, T(thick/n))
check("slab_focus_shift", thick-slab[0, 1], thick*(1-1/n))

# Match both finite-focal and equality cases; stable cavity fixed points.
z1, z2 = .1, .025
for f in [.05, .051, .1, .2]:
    for sign in [-1, 1]:
        root = np.sqrt(max(0., f*f-z1*z2))
        ell, lp = f+sign*np.sqrt(z1/z2)*root, f+sign*np.sqrt(z2/z1)*root
        check(f"waist_match_{f}_{sign}", prop(1j*z1, T(lp) @ lens(f) @ T(ell)), 1j*z2)
m = lens(.1) @ T(.1) @ lens(.1) @ T(.1)
q = -.05 + 1j*np.sqrt(3)*.05
check("negative_trace_stability", np.trace(m)/2, -.5)
check("negative_trace_fixed_point", prop(q, m), q)
check("confocal_degenerate_matrix", mirror(.2) @ T(.2) @ mirror(.2) @ T(.2), -np.eye(2))
check("plane_plane_jordan_growth", np.linalg.matrix_power(T(.2), 10), T(2.))
for j in range(40):
    # Half with positive g, half negative; exclude singular boundary.
    L = rng.uniform(.05, .8)
    g1, g2 = rng.uniform(.05, .9, 2) * (1 if j % 2 else -1)
    r1, r2 = L/(1-g1), L/(1-g2)
    dg = g1+g2-2*g1*g2
    zw = L*g2*(1-g1)/dg
    zr = np.sqrt(L*L*g1*g2*(1-g1*g2)/dg**2)
    q = -zw + 1j*zr
    m = mirror(r1) @ T(L) @ mirror(r2) @ T(L)
    check(f"cavity_fixed_point_{j}", prop(q, m), q)
    check(f"cavity_trace_{j}", np.trace(m)/2, 2*g1*g2-1)
    check(f"mirror1_curvature_{j}", (1/q).real, -1/r1)
    check(f"mirror2_curvature_{j}", (1/(q+L)).real, 1/r2)
    w_sq_fixed = -lam/(np.pi*(1/q).imag)
    check(f"fixed_point_spot_{j}", lam*abs(m[0, 1])/(np.pi*np.sqrt(1-(np.trace(m)/2)**2)), w_sq_fixed)
    alpha = np.arctan((L-zw)/zr)-np.arctan(-zw/zr)
    check(f"gouy_branch_{j}", np.cos(alpha), np.sign(g1)*np.sqrt(g1*g2))

# HG second moments by quadrature in x/w: <x^2>/w^2=(2m+1)/4.
for mode in range(5):
    intensity = lambda x: eval_hermite(mode, np.sqrt(2)*x)**2*np.exp(-2*x*x)
    norm = quad(intensity, -10, 10)[0]
    second = quad(lambda x: x*x*intensity(x), -10, 10)[0]/norm
    check(f"hg_second_moment_{mode}", second, (2*mode+1)/4)

# Small finite-difference check of paraxial PDE, nondimensional k=2,zR=1,w0=1.
def field(x, y, z):
    q = z+1j
    return 1j/q*np.exp(-1j*(x*x+y*y)/q)
delta = 1e-4
x, y, z = .3, -.2, .7
u = field(x, y, z)
lap = (field(x+delta,y,z)+field(x-delta,y,z)+field(x,y+delta,z)+field(x,y-delta,z)-4*u)/delta**2
dz = (field(x,y,z+delta)-field(x,y,z-delta))/(2*delta)
check("paraxial_pde_finite_difference", lap-4j*dz, 0, atol=2e-7)

w0, f = .0003, .1
zr = np.pi*w0*w0/lam
lp, zrp = waist(zr, f, 0)
L, R = .2, .3
zc = np.sqrt(L/2*(R-L/2))
numbers = {
    "wavelength_m": lam,
    "example_rayleigh_length_mm": zr*1e3,
    "example_half_divergence_mrad": lam/(np.pi*w0)*1e3,
    "example_focus_distance_mm": lp*1e3,
    "example_focused_waist_um": np.sqrt(lam*zrp/np.pi)*1e6,
    "example_back_focal_radius_um": lam*f/(np.pi*w0)*1e6,
    "cavity_waist_um": np.sqrt(lam*zc/np.pi)*1e6,
    "cavity_mirror_radius_um": np.sqrt(lam*zc/np.pi)*np.sqrt(1+(L/2/zc)**2)*1e6,
    "matching_input_waist_um": np.sqrt(lam*z1/np.pi)*1e6,
    "matching_target_waist_um": np.sqrt(lam*z2/np.pi)*1e6,
    "gain_threshold_per_m": .01+np.log(1/(.999*.95))/(2*.3),
    "gaussian_pulse_peak_MW": .5/(5e-9*pulse_integral)/1e6,
    "coherence_length_m_for_1MHz_Lorentz": c/(np.pi*1e6),
    "quantum_energy_ratio_808_1064": 808/1064,
    "hene_linewidth_Hz": [c/(632.8e-9)**2*dw*1e-9 for dw in [1e-9,1e-11]],
}

# 01: Generic level schemes; no physical energy scale.
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), layout="constrained")
for ax, levels, title in zip(axes, [[0, 2.3, 3.3], [0, .7, 2.3, 3.3]], ["三能级：下激光能级是基态", "四能级：下激光能级快速排空"]):
    for val in levels:
        ax.plot([.5, 3.6], [val,val], color="#3d4651", lw=2)
    def arrow(x, y0, y1, color, label, tx):
        ax.annotate("", (x,y1), (x,y0), arrowprops={"arrowstyle":"->","color":color,"lw":2.5})
        ax.text(tx, (y0+y1)/2, label, color=color, va="center", fontsize=13)
    arrow(.85, 0, 3.3, blue, "泵浦", .03)
    arrow(3.1, 3.3, 2.3, gold, "快弛豫", 3.25)
    arrow(2., 2.3, levels[1] if len(levels)==4 else 0, red, "激光", 2.15)
    if len(levels)==4:
        arrow(3.1,.7,0,gold,"快弛豫",3.25)
    ax.text(.55,2.43,"长寿命上能级", fontsize=12)
    ax.text(.55,-.25,"基态", fontsize=12)
    ax.set(xlim=(-.1,4.3), ylim=(-.5,3.8), title=title)
    ax.axis("off")
fig.suptitle("泵浦通道与激光通道分离，才可能维持反转", fontsize=17)
save(fig, "01-levels.png")

# 02: Coherence and pulse intensity, normalized axes.
fig, ax = plt.subplots(1,2,figsize=(12.8,4.8),layout="constrained")
t = np.linspace(-2,2,501)
ax[0].plot(t,np.exp(-np.pi*abs(t)),color=blue,lw=2.5)
ax[0].axhline(np.exp(-1),color=gold,ls="--",label=r"$1/e$")
ax[0].set(xlabel=r"$\Delta\nu\,\tau$",ylabel=r"$|g^{(1)}(\tau)|$",title="Lorentz 光谱的一阶相干性")
ax[0].legend()
ax[1].plot(t,np.exp(-4*np.log(2)*t*t),color=red,lw=2.5)
ax[1].axhline(.5,color=gold,ls="--")
ax[1].set(xlabel=r"$t/\tau_p$",ylabel=r"$P(t)/P_{\rm pk}$",title="Gaussian 脉冲强度；FWHM 为 1")
for a in ax: a.grid(alpha=.2)
save(fig,"02-coherence.png")

# 03: Radius, curvature and Gouy.
fig, ax = plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
s = np.linspace(-4,4,601)
ax[0].plot(s,np.sqrt(1+s*s),color=blue,lw=2.5)
ax[0].plot(s,-np.sqrt(1+s*s),color=blue,lw=2.5)
ax[0].fill_between(s,-np.sqrt(1+s*s),np.sqrt(1+s*s),alpha=.1,color=blue)
ax[0].set(ylabel=r"$r/w_0$", title="光束包络")
ax[1].plot(s,s/(1+s*s),color=green,lw=2.5)
ax[1].set(ylabel=r"$z_R/R$",title="曲率倒数")
ax[2].plot(s,np.arctan(s)/np.pi,color=red,lw=2.5)
ax[2].set(ylabel=r"$\psi/\pi$",title="Gouy 相位")
for a in ax:
    a.set_xlabel(r"$z/z_R$")
    a.axvline(0,color="gray",lw=.7)
    a.axvline(1,color=gold,ls="--",alpha=.6)
    a.axvline(-1,color=gold,ls="--",alpha=.6)
    a.grid(alpha=.2)
save(fig,"03-gaussian-geometry.png")

# 04: Computed HG and LG intensity patterns.
xg = np.linspace(-2.6,2.6,351)
xx, yy = np.meshgrid(xg,xg)
rr2 = xx*xx+yy*yy
fig, ax = plt.subplots(2,3,figsize=(11,7.4),layout="constrained")
fields = []
for m,n in [(0,0),(1,0),(1,1)]:
    fields.append((f"HG({m},{n})",eval_hermite(m,np.sqrt(2)*xx)*eval_hermite(n,np.sqrt(2)*yy)*np.exp(-rr2)))
for p,ell in [(0,1),(1,0),(1,2)]:
    fields.append((f"LG(p={p}, l={ell})",(np.sqrt(2*rr2)**abs(ell))*eval_genlaguerre(p,abs(ell),2*rr2)*np.exp(-rr2)))
for a,(label,ef) in zip(ax.flat,fields):
    intensity = abs(ef)**2
    a.imshow(intensity/intensity.max(),origin="lower",extent=[-2.6,2.6,-2.6,2.6],cmap="magma",vmin=0,vmax=1)
    a.set(title=label,xlabel="x/w",ylabel="y/w",xticks=[-2,0,2],yticks=[-2,0,2])
save(fig,"04-transverse-modes.png")

# 05: exact lens transforms, fixed focal-plane spot.
fig, ax = plt.subplots(1,2,figsize=(12.8,4.8),layout="constrained")
ell = np.linspace(0,.4,600)
lp,zrp = waist(.05,.1,ell)
ax[0].plot(ell*1e3,lp*1e3,color=blue,lw=2.5,label="新束腰位置")
ax[0].axhline(100,color=red,ls="--",label="后焦平面")
ax[0].set(ylabel="透镜后距离 / mm",title="束腰不一定在后焦平面")
ax[1].plot(ell*1e3,np.sqrt(zrp/.05),color=green,lw=2.5,label=r"$w_0'/w_0$")
ax[1].axhline(2,color=red,ls="--",label=r"$w(f)/w_0$")
ax[1].set(ylabel="半径比",title="固定 f 与输入束腰，移动透镜")
for a in ax:
    a.axvline(100,color=gold,ls=":")
    a.set_xlabel("旧束腰到透镜距离 l / mm")
    a.legend()
    a.grid(alpha=.2)
save(fig,"05-lens-transform.png")

# 06: normalized spatial overlap.
ratios = np.linspace(.25,3,400)
mis = np.linspace(-4,4,401)
rr,mm = np.meshgrid(ratios,mis)
eta = 4/((rr+1/rr)**2+mm*mm)
fig, ax = plt.subplots(figsize=(9.6,6),layout="constrained")
im=ax.pcolormesh(rr,mm,eta,shading="auto",cmap="viridis",vmin=0,vmax=1)
cs=ax.contour(rr,mm,eta,levels=[.2,.4,.6,.8,.95],colors="white",linewidths=.9)
ax.clabel(cs,fmt="%.2f",fontsize=10)
ax.plot([1],[0],"r+",ms=13,mew=2)
ax.set(xlabel=r"$w_1/w_2$",ylabel=r"$(\pi w_1w_2/\lambda)(1/R_1-1/R_2)$",title="同轴、同偏振基模的功率重叠效率")
fig.colorbar(im,ax=ax,label=r"$\eta$")
save(fig,"06-mode-overlap.png")

# 07: g-parameter map and symmetric beam radius.
g = np.linspace(-2,2,700)
gx,gy = np.meshgrid(g,g)
stable = (gx*gy>0)&(gx*gy<1)
fig, ax=plt.subplots(1,2,figsize=(12.8,5.2),layout="constrained")
ax[0].contourf(gx,gy,stable.astype(float),levels=[-.1,.5,1.1],colors=["#f1f4f8","#72adce"])
ax[0].contour(gx,gy,gx*gy,levels=[1],colors=[blue],linewidths=1.2)
ax[0].axvline(0,color="gray",ls="--")
ax[0].axhline(0,color="gray",ls="--")
ax[0].set(xlabel=r"$g_1$",ylabel=r"$g_2$",title=r"蓝色：$0<g_1g_2<1$",aspect="equal")
t = np.linspace(.01,1.99,600) # L/R, R fixed; w0 / sqrt(lambda R/pi)
wr = (t/2*(1-t/2))**.25
ax[1].plot(t,wr,color=green,lw=2.5)
ax[1].plot([1],[np.sqrt(.5)],marker="o",mfc="white",mec=red,ms=9,ls="")
ax[1].annotate("共焦：特殊退化点",xy=(1,np.sqrt(.5)),xytext=(1.03,.48),
               arrowprops={"arrowstyle":"->","color":red},color=red)
ax[1].set(xlabel="L/R（两镜曲率相等）",ylabel=r"$w_0/\sqrt{\lambda R/\pi}$",title="常用对称高斯模式分支")
ax[1].grid(alpha=.2)
save(fig,"07-cavity-stability.png")

# 08: second-moment caustics; M² labels are the propagation ratio.
fig,ax=plt.subplots(figsize=(9.5,5.4),layout="constrained")
zz=np.linspace(-2.5,2.5,600)
for factor,color in [(1,blue),(1.5,green),(2,red)]:
    ww=np.sqrt(1+(factor*zz)**2)
    ax.plot(zz,ww,color=color,lw=2.5,label=f"$M^2={factor}$")
ax.set(xlabel=r"$z/(\pi w_{\sigma0}^2/\lambda)$",ylabel=r"$w_\sigma(z)/w_{\sigma0}$",
       title="同一波长、同一束腰二阶矩半径")
ax.grid(alpha=.2)
ax.legend()
save(fig,"08-m2-propagation.png")

# 09: GRIN harmonic rays over one pitch.
fig,ax=plt.subplots(figsize=(11,4.8),layout="constrained")
phase=np.linspace(0,2*np.pi,600)
for y0,slope,color in [(1,0,blue),(.5,.7,green),(0,1,red),(-.8,.3,gold)]:
    ax.plot(phase/(2*np.pi),y0*np.cos(phase)+slope*np.sin(phase),color=color,lw=2.2,
            label=fr"$y_0/a={y0},\ \theta_0/(\gamma a)={slope}$")
for v in [.25,.5,.75]:
    ax.axvline(v,color="gray",ls="--",alpha=.4)
ax.set(xlabel=r"传播距离 / 节距 $=z\gamma/(2\pi)$",ylabel=r"$y/a$",title="GRIN 近轴模型：光线连续作正弦振荡")
ax.legend(fontsize=10,ncol=2,loc="upper right")
ax.grid(alpha=.2)
save(fig,"09-grin-rays.png")

report={"description":"Independent numerical checks for the laser-course article series",
        "seed":20261004,"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "checks_total":len(checks),"checks_passed":sum(x["passed"] for x in checks),
        "illustrative_values":numbers,"checks":checks,
        "figures":[p.name for p in sorted(figdir.glob("*.png"))]}
(out/"verification-results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="checks"},ensure_ascii=False,indent=2))
