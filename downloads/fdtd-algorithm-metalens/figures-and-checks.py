"""FDTD tutorial: deterministic figures and a real 1-D periodic Yee solver.

Run: python figures-and-checks.py --output-dir PATH
Requires numpy and matplotlib. SI units are used only in the engineering
estimates; the executable 1-D benchmark uses c=epsilon=mu=1, domain length 1.
No figure is a full-device metalens FDTD simulation or an experimental result.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'figures')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False, 'font.size':12, 'figure.facecolor':'#faf8f2',
    'axes.facecolor':'#faf8f2','text.color':'#20334b','axes.labelcolor':'#20334b',
    'axes.spines.top':False,'axes.spines.right':False})
TEAL, ORANGE, BLUE = '#087f8c', '#cc7921', '#547db3'

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def periodic_mode(n, final_time=.37, max_courant=.5, mode=2):
    """E_x(z), H_y(z): both updates have minus signs; periodic boundaries.

    E[j] lives at z=j*dz, H[j] at z=(j+1/2)*dz.
    Start with exact continuum standing wave at E time 0 and H time -dt/2.
    Return E at final_time and its continuum reference.
    """
    dz = 1/n
    steps = int(np.ceil(final_time / (max_courant*dz)))
    dt = final_time/steps
    s = dt/dz
    z = np.arange(n)*dz
    k = 2*np.pi*mode
    e = np.sin(k*z)
    h = np.cos(k*(z+dz/2))*np.sin(k*dt/2)
    for _ in range(steps):
        h -= s*(np.roll(e,-1)-e)
        e -= s*(h-np.roll(h,1))
    exact = np.sin(k*z)*np.cos(k*final_time)
    error = float(np.sqrt(np.mean((e-exact)**2)))
    return z, e, exact, error

# 1. Actual TM_z spatial staggering, with a separate temporal diagram.
fig, (ax, at) = plt.subplots(1,2,figsize=(12.4,4.8),layout='constrained')
for i in range(3):
    ax.axvline(i,color='#d8dfe3',lw=1)
    ax.axhline(i,color='#d8dfe3',lw=1)
for i in range(3):
    for j in range(3):
        ax.scatter(i,j,s=95,color=TEAL,zorder=4)
for i in range(3):
    for j in range(2):
        ax.plot([i-.14,i+.14],[j+.5,j+.5],color=ORANGE,lw=4)
for i in range(2):
    for j in range(3):
        ax.plot([i+.5,i+.5],[j-.14,j+.14],color=BLUE,lw=4)
ax.scatter([],[],s=65,color=TEAL,label=r'$E_z(i,j)$（出平面）')
ax.plot([],[],color=ORANGE,lw=4,label=r'$H_x(i,j+1/2)$')
ax.plot([],[],color=BLUE,lw=4,label=r'$H_y(i+1/2,j)$')
ax.add_patch(Rectangle((.5,.5),1,1,fill=False,ls='--',lw=1.7,color=TEAL))
ax.set(xlim=(-.4,2.4),ylim=(-.4,2.4),xlabel=r'$x / \Delta x$',ylabel=r'$y / \Delta y$',title='空间：电场与磁场相隔半个网格')
ax.set_aspect('equal'); ax.legend(loc='upper left',bbox_to_anchor=(-.13,-.19),fontsize=10,ncol=1)
at.axhline(1,color='#b8c4cb'); at.axhline(0,color='#b8c4cb')
for t in [0,1,2]:
    at.scatter(t,1,s=90,color=TEAL)
    at.text(t,1.16,rf'$E^{{{["n","n+1","n+2"][t]}}}$',ha='center')
for t, label in [(-.5,'n-1/2'),(.5,'n+1/2'),(1.5,'n+3/2')]:
    at.scatter(t,0,s=90,color=ORANGE)
    at.text(t,-.27,rf'$H^{{{label}}}$',ha='center')
for a,b in [((0,.9),(.5,.1)),((.5,.1),(1,.9)),((1,.9),(1.5,.1)),((1.5,.1),(2,.9))]:
    at.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=13,color=BLUE,lw=1.5))
at.text(.65,-.7,'先由 E 更新 H，再由新 H 更新 E',ha='center',fontsize=12)
at.set(xlim=(-.85,2.3),ylim=(-.85,1.65),title='时间：蛙跳推进，相差半个时间步')
at.axis('off')
save(fig,'01-yee-grid')

# 2. Exact discrete dispersion relation, at a specified frequency.
fig,(ax,ay)=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
ppw=np.linspace(5,60,400)
for s in [.5,.9]:
    omega_dt=2*np.pi*s/ppw
    k_dx=2*np.arcsin(np.sin(omega_dt/2)/s)
    ratio=(2*np.pi/ppw)/k_dx
    ax.plot(ppw,ratio,label=rf'1D：$S={s}$',lw=2.3)
ax.axhline(1,color='#9da8af',ls='--'); ax.legend()
ax.set(xlabel='每个物理波长的网格数 Nλ',ylabel=r'$v_{\rm num}/c$',title='稳定的计算仍然存在相位误差')
theta=np.linspace(0,np.pi/2,181); N=10; s=.5
lhs=(np.sin(np.pi*s/N)/s)**2
lo=np.zeros_like(theta); hi=np.full_like(theta,1.5)
for _ in range(60):
    mid=(lo+hi)/2
    val=np.sin(mid*np.cos(theta)/2)**2+np.sin(mid*np.sin(theta)/2)**2
    lo=np.where(val<lhs,mid,lo); hi=np.where(val>=lhs,mid,hi)
ratio=(2*np.pi/N)/((lo+hi)/2)
ay.plot(np.degrees(theta),ratio,color=TEAL,lw=2.4)
ay.set(xlabel='传播方向与 x 轴夹角 / 度',ylabel=r'$v_{\rm num}/c$',title='2D 方格网格：Nλ = 10，S = 0.5')
ay.grid(alpha=.2)
save(fig,'02-numerical-dispersion')

# 3. Real solver convergence, against an independent continuum solution.
ns=np.array([40,80,160,320])
errors=np.array([periodic_mode(int(n))[3] for n in ns])
orders=np.log2(errors[:-1]/errors[1:])
assert np.all((orders>1.9)&(orders<2.1)), orders
fig,(ax,ay)=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
z,e,exact,_=periodic_mode(40)
zz=np.linspace(0,1,800)
ax.plot(zz,np.sin(4*np.pi*zz)*np.cos(4*np.pi*.37),color=TEAL,lw=2,label='连续方程解析解')
ax.plot(z,e,'o',ms=4,color=ORANGE,label='实际 Yee 迭代：40 格')
ax.set(xlabel='z / L',ylabel='归一化 Ex',title='周期驻波基准：t = 0.37 L/c'); ax.legend()
ay.loglog(ns,errors,'o-',color=TEAL,lw=2,label='实测 RMS 误差')
ay.loglog(ns,errors[0]*(ns[0]/ns)**2,'--',color=ORANGE,label='二阶参考线')
ay.set(xlabel='网格数 N',ylabel='RMS 误差',title='网格加密一倍，误差约缩小至 1/4')
ay.legend(); ay.grid(which='both',alpha=.2)
save(fig,'03-solver-convergence')

# 4. Distinct boundary problems; diagram dimensions are schematic.
fig,(ax,ay)=plt.subplots(1,2,figsize=(12.4,5.4),layout='constrained')
for a,title in [(ax,'单元库：无限周期阵列中的一个单元'),(ay,'整片器件：有限口径与开放空间')]:
    a.set(xlim=(0,10),ylim=(0,10),title=title)
    a.add_patch(Rectangle((.8,.8),8.4,8.4,fill=False,ec='#8fa0ad',lw=1.5))
    a.add_patch(Rectangle((.8,2.8),8.4,1.4,fc='#cddfe4',ec='none'))
    a.text(5,3.45,'基底',ha='center')
    a.plot([1.4,8.6],[2.1,2.1],color=BLUE,lw=2)
    a.text(5,1.55,'入射源 ↑',ha='center',color=BLUE)
    a.plot([1.4,8.6],[6.6,6.6],color=TEAL,ls='--',lw=2)
    a.text(5,7.05,'复数场 / 功率监视面',ha='center',fontsize=11,color=TEAL)
    for y in [.8,8.6]:
        a.add_patch(Rectangle((.8,y),8.4,.6,fc='#e8bb8a',alpha=.7))
    a.text(5,8.8,'PML',ha='center',va='center',fontsize=10)
    a.axis('off')
ax.add_patch(Rectangle((4,4.2),2,1.55,fc=TEAL))
for x in [.8,9.2]:
    ax.plot([x,x],[1.4,8.6],color=BLUE,lw=3,ls='--')
ax.text(.15,5.2,'周\n期',va='center',color=BLUE)
ax.text(9.45,5.2,'周\n期',va='center',color=BLUE)
ax.text(5,.1,'侧面相位由周期 / Bloch 条件连接',ha='center',fontsize=11)
for x,w in [(2.7,.5),(3.65,.68),(4.75,.9),(6,.68),(7.1,.5)]:
    ay.add_patch(Rectangle((x,4.2),w,1.55,fc=TEAL))
for x in [.8,8.6]: ay.add_patch(Rectangle((x,1.4),.6,7.2,fc='#e8bb8a',alpha=.7))
ay.text(5,.1,'侧面使用 PML，保留孔径边缘与相邻耦合',ha='center',fontsize=11)
save(fig,'04-boundary-models')

# 5. Analytic target phase. This is not a metalens full-wave calculation.
wl=.532; f=10.; R=5.; r=np.linspace(0,R,800)
phase=-2*np.pi/wl*(np.sqrt(f*f+r*r)-f)
fig,(ax,ay)=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
ax.plot(r,phase/(2*np.pi),color=TEAL,lw=2.4,label='连续目标相位 / 2π')
ax.plot(r,np.mod(phase,2*np.pi)/(2*np.pi),color=ORANGE,lw=1.7,label='包裹至 [0, 2π) 后 / 2π')
ax.set(xlabel='半径 r / μm',ylabel='相位 / 2π',title=r'D = 10 μm，f = 10 μm，$\lambda_0$ = 532 nm'); ax.legend()
for x in [-5,-3,-1,1,3,5]:
    ay.plot([x,0],[0,10],color=ORANGE,alpha=.7,lw=1.3)
ay.plot([-5,5],[0,0],color=TEAL,lw=6)
ay.scatter(0,10,color=ORANGE,s=70,zorder=5)
ay.annotate('目标焦点',(0,10),(1.2,10.5),arrowprops={'arrowstyle':'->','color':BLUE})
ay.text(-4.5,1,'平面相位元件',color=TEAL)
ay.set(xlim=(-6,6),ylim=(-.5,12),xlabel='x / μm',ylabel='z / μm',title='目标等光程示意（非 FDTD 场图）')
save(fig,'05-metalens-phase')

c=299792458.
dx=20e-9
dt=.95*dx/(c*np.sqrt(3))
cells=600*600*400
report={'benchmark':'1D periodic Yee standing wave, c=epsilon=mu=1, L=1, mode=2, t=0.37',
    'grids':ns.tolist(),'rms_errors':errors.tolist(),'observed_orders':orders.tolist(),
    'checks_passed':{'all_finite':bool(np.all(np.isfinite(errors))),'second_order':True},
    'engineering_example':{'dx_nm':20,'dt_attoseconds':dt*1e18,'steps_for_200fs':int(np.ceil(200e-15/dt)),
        'cells':cells,'six_float32_fields_GB':cells*6*4/1e9,'NA':R/np.sqrt(R*R+f*f),
        'edge_phase_rad':float(phase[-1]),'edge_optical_path_um':float(np.sqrt(f*f+R*R)-f),
        'airy_FWHM_um':float(.514*wl/(R/np.sqrt(R*R+f*f)))},
    'scope':'Analytic teaching figures plus a 1D solver benchmark; no actual metalens FDTD results.'}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
