"""Reproduce this article's paraxial diagrams and numerical checks.

Dependencies: numpy, matplotlib. Run from any directory; lengths are in mm.
Generated cover 00-cover.png is independent of this script.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "source/img/ray-transfer-matrix"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#faf8f2", "axes.facecolor": "#faf8f2", "text.color": "#20334b", "axes.labelcolor": "#20334b", "svg.fonttype": "none"})
TEAL, AMBER, BLUE = "#087f8c", "#d58929", "#527fbd"

def T(d): return np.array([[1., d], [0., 1.]])
def L(f): return np.array([[1., 0.], [-1/f, 1.]])
def S(n1, n2, r): return np.array([[1., 0.], [(n1-n2)/(n2*r), n1/n2]])
def mirror(r): return np.array([[1., 0.], [-2/r, 1.]])
def qmap(m, q): return (m[0, 0]*q + m[0, 1])/(m[1, 0]*q + m[1, 1])
def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)
def axis(ax, xmin, xmax):
    ax.axhline(0, color="#8695a1", lw=1, ls="--")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("沿光轴的位置 z / mm")
    ax.set_ylabel("光线高度 y / mm")
def lens(ax, x, h):
    ax.annotate("", (x, h), (x, -h), arrowprops={"arrowstyle": "<->", "color": BLUE, "lw": 2.6})

# 1. Both diagrams are computed from the same ray vector.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
ax = axes[0]
axis(ax, 0, 100)
ax.plot([0, 100], [1, 3], color=TEAL, lw=2.8)
ax.scatter([0, 100], [1, 3], color=TEAL, zorder=5)
ax.set_ylim(-.3, 4.5)
ax.text(8, 3.7, "高度改变，倾角不变\ny2 = y1 + dθ1", fontsize=14)
ax.text(20, .3, "d = 100 mm；θ = 0.020 rad")
ax.set_title("均匀介质传播 T(d)", loc="left", weight="bold")
ax = axes[1]
axis(ax, -60, 150)
ax.set_ylim(-3.2, 3.8)
lens(ax, 0, 2.8)
for y, c in [(2, TEAL), (-1, AMBER)]:
    ax.plot([-60, 0, 150], [y, y, y*(1-150/100)], color=c, lw=2.5)
ax.scatter([100], [0], color="#20334b", zorder=5)
ax.text(105, .2, "F")
ax.text(16, 2.45, "高度不变，倾角改变\nθ2 = θ1 − y1/f", fontsize=14)
ax.set_title("薄透镜 L(f)，f = 100 mm", loc="left", weight="bold")
save(fig, "01-basic-elements")

# 2. A family of rays from one object point reaches the same image point.
fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
axis(ax, -160, 320)
lens(ax, 0, 6)
ax.axvline(-150, color="#8695a1", ls=":")
ax.axvline(300, color="#8695a1", ls=":")
for th, c in zip([-.02, 0, .02], [BLUE, TEAL, AMBER]):
    r0=np.array([2., th]); r1=T(150)@r0; r2=T(300)@L(100)@r1
    ax.plot([-150, 0, 300], [2, r1[0], r2[0]], color=c, lw=2.4)
ax.annotate("", (-150, 2), (-150, 0), arrowprops={"arrowstyle":"->", "lw":3, "color":"#20334b"})
ax.annotate("", (300, -4), (300, 0), arrowprops={"arrowstyle":"->", "lw":3, "color":"#20334b"})
ax.set_ylim(-6.5, 7)
ax.text(-145, 5.7, "物点 y = 2 mm")
ax.text(125, -5.7, "像点 y′ = −4 mm；m = −2")
ax.text(25, 5.7, "f = 100 mm；s = 150 mm；s′ = 300 mm")
ax.set_title("B = 0：同一物点发出的近轴光线，在像面重新相交", loc="left", weight="bold")
save(fig, "02-imaging")

# 3. 4f relay from front focal plane to back focal plane.
fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
axis(ax, -15, 615)
lens(ax, 100, 6); lens(ax, 400, 6)
for z in [0, 200, 600]: ax.axvline(z, color="#8695a1", ls=":")
for th,c in zip([-.02, 0, .02], [BLUE, TEAL, AMBER]):
    r=np.array([2., th]); yy=[r[0]]
    for d, f in [(100,100), (300,200)]:
        r=T(d)@r; yy.append(r[0]); r=L(f)@r
    r=T(200)@r; yy.append(r[0])
    ax.plot([0,100,400,600], yy, color=c, lw=2.4)
ax.set_ylim(-8.8,8)
for z, text in [(0,"物面"),(100,"L1"),(200,"共同焦平面"),(400,"L2"),(600,"像面")]:
    ax.text(z,6.7,text,ha="center")
ax.text(230,-8,"f1 = 100 mm；f2 = 200 mm；m = −2")
ax.set_title("4f 中继：参考面选在第一片前焦面与第二片后焦面", loc="left", weight="bold")
save(fig,"03-4f-relay")

# 4. Gaussian beam waist initially lies in the thin-lens plane.
lam=.0006328; w0=.3; f=100.; zr=np.pi*w0*w0/lam
qout=qmap(L(f),1j*zr); zfocus=-qout.real; zrnew=qout.imag
wnew=np.sqrt(lam*zrnew/np.pi)
fig,ax=plt.subplots(figsize=(11,4.6),layout="constrained")
zz=np.linspace(0,180,800); qq=qout+zz
ww=np.sqrt(-lam/(np.pi*np.imag(1/qq)))
ax.fill_between(zz,-ww,ww,color=TEAL,alpha=.14)
ax.plot(zz,ww,color=TEAL,lw=2.3,label="高斯光束 1/e² 强度半径包络")
ax.plot(zz,-ww,color=TEAL,lw=2.3)
ax.plot([0,100,180],[.3,0,.24],ls="--",color=AMBER,label="几何光线包络参考")
ax.plot([0,100,180],[-.3,0,-.24],ls="--",color=AMBER)
lens(ax,0,.34)
ax.axvline(zfocus,color=TEAL,ls=":")
ax.axvline(100,color=AMBER,ls=":")
ax.text(105,.26,f"新束腰：{zfocus:.2f} mm\n最小半径：{wnew*1000:.2f} μm")
axis(ax,-5,180)
ax.set_ylabel("横向位置 / mm")
ax.set_ylim(-.36,.42)
ax.legend(loc="upper right",fontsize=10)
ax.set_title("高斯光束聚焦：最小光斑不等于几何焦点",loc="left",weight="bold")
save(fig,"04-gaussian-focus")

# 5. Strict stability interior and explicitly excluded boundaries.
fig,ax=plt.subplots(figsize=(7.5,6),layout="constrained")
gg=np.linspace(-2.5,2.5,801); gx,gy=np.meshgrid(gg,gg)
stable=(gx*gy>0)&(gx*gy<1)
ax.contourf(gx,gy,stable.astype(float),levels=[.5,1.5],colors=["#b8deda"])
for x in [np.linspace(.4,2.5,600),np.linspace(-2.5,-.4,600)]:
    ax.plot(x,1/x,color=TEAL,lw=1.5,ls="--")
ax.axhline(0,color="#83939b",ls="--",lw=1)
ax.axvline(0,color="#83939b",ls="--",lw=1)
ax.scatter([1/3],[1/3],color=AMBER,s=70,zorder=6)
ax.annotate("本文算例 (1/3, 1/3)",(1/3,1/3),(1.05,.7),arrowprops={"arrowstyle":"->","color":AMBER},fontsize=11)
ax.text(-2.28,-.38,"严格稳定区",fontsize=13)
ax.text(-2.28,1.95,"着色区域：0 < g1g2 < 1\n虚线边界须单独分析",fontsize=13)
ax.set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel="g1 = 1 − L/R1",ylabel="g2 = 1 − L/R2")
ax.set_aspect("equal")
ax.set_title("双球面镜腔的稳定图",loc="left",weight="bold")
save(fig,"05-cavity-stability")

# Independent checks: exact Snell refraction, direct ray geometry, closed forms.
checks={}
mi=T(300)@L(100)@T(150)
np.testing.assert_allclose(mi,[[-2,0],[-.01,-.5]],atol=1e-12)
for th in np.linspace(-.02,.02,21):
    height_at_lens=2+150*th
    image_height=height_at_lens+300*(th-height_at_lens/100)
    np.testing.assert_allclose(image_height,-4,atol=1e-12)
checks["imaging_matrix"]=mi.tolist()
m4=T(200)@L(200)@T(300)@L(100)@T(100)
np.testing.assert_allclose(m4,[[-2,0],[0,-.5]],atol=1e-12)
checks["relay_matrix"]=m4.tolist()
mt=S(1.5,1,-50)@T(5)@S(1,1.5,50)
fe=1/((1.5-1)*(1/50-1/(-50)+(1.5-1)*5/(1.5*50*(-50))))
np.testing.assert_allclose(-1/mt[1,0],fe)
np.testing.assert_allclose(np.linalg.det(mt),1)
checks["thick_lens"]={"matrix":mt.tolist(),"efl_mm":fe,"bfl_mm":-mt[0,0]/mt[1,0]}
for n1,n2,r in [(1,1.5,50),(1.5,1,-50),(1,1.7,-90)]:
    h=.0001; th=.000001
    normal=-np.arcsin(h/r)
    exact=normal+np.arcsin(n1/n2*np.sin(th-normal))
    approx=(S(n1,n2,r)@np.array([h,th]))[1]
    np.testing.assert_allclose(exact,approx,rtol=1e-9,atol=1e-15)
np.testing.assert_allclose(zfocus,f*zr*zr/(f*f+zr*zr))
np.testing.assert_allclose(wnew,w0*f/np.sqrt(f*f+zr*zr))
checks["gaussian"]={"input_rayleigh_mm":zr,"q_after_lens_mm":[qout.real,qout.imag],"waist_position_mm":zfocus,"waist_radius_um":wnew*1000}
length=200.; radius=300.; mr=mirror(radius)@T(length)@mirror(radius)@T(length)
q=-length/2+1j*np.sqrt(length/2*(radius-length/2))
np.testing.assert_allclose(qmap(mr,q),q)
np.testing.assert_allclose(np.trace(mr)/2,2*(1-length/radius)**2-1)
conf=mirror(200)@T(200)@mirror(200)@T(200)
np.testing.assert_allclose(conf,-np.eye(2),atol=1e-12)
checks["cavity"]={"matrix":mr.tolist(),"half_trace":np.trace(mr)/2,"q_mm":[q.real,q.imag],"waist_radius_mm":np.sqrt(lam*q.imag/np.pi)}
checks["status"]="All analytical and numerical checks passed."
(Path(__file__).parent/"validation.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(checks,ensure_ascii=False,indent=2))
