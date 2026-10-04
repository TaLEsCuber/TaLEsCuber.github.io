"""Reproduce this article's figures and numerical checks. Python + NumPy/SciPy/Matplotlib."""
from pathlib import Path
import json
import numpy as np
from scipy.special import j1, jn_zeros
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'img' / 'numerical-aperture'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
    'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#f6f8fc', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#f6f8fc'})
BLUE, ORANGE, GREEN = '#2463a6', '#d87528', '#198579'
def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
def airy(u):
    u=np.asarray(u, dtype=float)
    out=np.ones_like(u)
    np.divide(2*j1(u), u, out=out, where=u!=0)
    return out**2

# 1. Geometric cones: all panels use the same spatial scale.
fig, axs=plt.subplots(1,3,figsize=(13,4.6))
for ax, na in zip(axs,[.25,.5,.8]):
    theta=np.arcsin(na); height=2*np.tan(theta)
    ax.add_patch(Polygon([[0,0],[2,height],[2,-height]],color=BLUE,alpha=.14))
    ax.plot([0,2],[0,height],color=BLUE); ax.plot([0,2],[0,-height],color=BLUE)
    ax.plot([2,2],[-height,height],color=GREEN,lw=5)
    ax.axhline(0,color='#8994a5',lw=1,ls='--'); ax.scatter([0],[0],color=ORANGE,s=55,zorder=4)
    ax.add_patch(Arc((0,0),1.3,1.3,theta1=0,theta2=np.degrees(theta),color=ORANGE,lw=2))
    ax.text(.75,.2,r'$\theta$',color=ORANGE,fontsize=17)
    ax.text(-.12,-.5,'点光源',ha='center'); ax.text(2.1,0,'孔径',va='center')
    ax.set_title(f'NA = {na:.2f}\n半角 {np.degrees(theta):.2f}°',pad=14)
    ax.set_xlim(-.4,2.8); ax.set_ylim(-3,3); ax.set_aspect('equal'); ax.axis('off')
fig.subplots_adjust(top=.75,bottom=.14)
fig.suptitle('同一介质中，NA 越大，接收的光锥越宽',fontsize=19,y=.99)
fig.text(.5,.02,'空气 n = 1；NA = n sin θ；完整锥角为 2θ。绿色线段表示有效孔径。',ha='center')
save(fig,'01-cones.png')

# 2. Index and spatial frequency.
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
deg=np.linspace(0,85,250)
for n,label,col in [(1,'空气 n = 1.000',BLUE),(1.333,'水 n = 1.333',GREEN),(1.515,'浸油 n = 1.515',ORANGE)]:
    axs[0].plot(deg,n*np.sin(np.radians(deg)),label=label,color=col,lw=2.5)
axs[0].set(xlabel='半角 θ / °',ylabel='NA',title='相同光锥角，更高折射率允许更高 NA')
axs[0].legend(); axs[0].grid(alpha=.18)
t=np.linspace(0,2*np.pi,300)
for na,col in [(.8,BLUE),(.4,ORANGE)]:
    axs[1].fill(na*np.cos(t),na*np.sin(t),color=col,alpha=.18)
    axs[1].plot(na*np.cos(t),na*np.sin(t),color=col,label=f'NA = {na}')
axs[1].set(xlabel=r'$k_x/k_0$',ylabel=r'$k_y/k_0$',title=r'横向波矢支持域：$k_\perp \leq k_0\,NA$')
axs[1].set_aspect('equal'); axs[1].legend(loc='upper right'); axs[1].grid(alpha=.18)
save(fig,'02-index-spectrum.png')

# 3. PSF width and Airy rings; scalar uniform pupil, same peak.
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
x=np.linspace(-2,2,1500); lam=.55
for na,col in [(.25,ORANGE),(.5,BLUE),(.8,GREEN)]:
    axs[0].plot(x,airy(2*np.pi*na*x/lam),lw=2,label=f'NA = {na}',color=col)
axs[0].set(xlabel='物方横向位置 / μm',ylabel='归一化强度（各自峰值为 1）',title='550 nm：NA 越大，点扩散函数越窄')
axs[0].legend(); axs[0].grid(alpha=.18)
xx,yy=np.meshgrid(np.linspace(-1.5,1.5,501),np.linspace(-1.5,1.5,501))
im=axs[1].imshow(np.log10(np.maximum(airy(2*np.pi*.5*np.hypot(xx,yy)/lam),1e-4)),extent=[-1.5,1.5,-1.5,1.5],cmap='magma',vmin=-4,vmax=0)
axs[1].set(xlabel='x / μm',ylabel='y / μm',title='NA = 0.50 的艾里图样（对数色标）')
fig.colorbar(im,ax=axs[1],label='log10(I / I0)')
save(fig,'03-airy.png')

# 4. Incoherent two-point image.
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
x=np.linspace(-1.8,1.8,1400); na=.5; dr=jn_zeros(1,1)[0]*lam/(2*np.pi*na)
for ax,fac in zip(axs,[.65,1,1.5]):
    d=fac*dr; left=airy(2*np.pi*na*(x+d/2)/lam); right=airy(2*np.pi*na*(x-d/2)/lam)
    ax.plot(x,left,'--',color=BLUE,alpha=.6); ax.plot(x,right,'--',color=ORANGE,alpha=.6)
    ax.plot(x,left+right,color=GREEN,lw=2.5)
    ax.set(xlabel='位置 / μm',ylabel='相对强度',title=f'间距 = {fac:.2f} × 瑞利间距',ylim=(0,1.8))
    ax.grid(alpha=.18)
fig.suptitle('两个等强、互不相干点源：强度相加；实线为总强度',fontsize=17)
save(fig,'04-rayleigh.png')

# 5. Collection fraction integrated over a sphere.
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
na=np.linspace(0,.95,400); eta=(1-np.sqrt(1-na**2))/2
axs[0].plot(na,100*eta,lw=2.5,color=BLUE,label='精确立体角')
axs[0].plot(na,25*na**2,'--',color=ORANGE,label='小 NA 近似：NA² / 4')
axs[0].set(xlabel='空气中的 NA',ylabel='占全空间发射的比例 / %',title='各向同性点源，单侧收光')
axs[0].legend(); axs[0].grid(alpha=.18)
vals=[.1,.25,.5,.7,.9]; pct=100*(1-np.sqrt(1-np.array(vals)**2))/2
axs[1].bar([str(v) for v in vals],pct,color=[BLUE]*5)
for i,v in enumerate(pct): axs[1].text(i,v+.5,f'{v:.2f}%',ha='center')
axs[1].set(xlabel='NA',ylabel='几何收光率 / %',title='NA = 0.50 仅收集全空间约 6.70%',ylim=(0,33))
save(fig,'05-collection.png')

# 6. Gaussian focusing and longitudinal scale.
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
z=np.linspace(-45,45,700); lam=.532
for na,col in [(.1,BLUE),(.2,ORANGE)]:
    w0=lam/(np.pi*na); zr=np.pi*w0*w0/lam; w=w0*np.sqrt(1+(z/zr)**2)
    axs[0].plot(z,w,color=col,label=f'NA_eff = {na:.1f}，w0 = {w0:.2f} μm')
    axs[0].plot(z,-w,color=col); axs[0].fill_between(z,-w,w,color=col,alpha=.09)
axs[0].set(xlabel='距束腰的轴向距离 z / μm',ylabel='1/e² 强度半径边界 / μm',title='近轴高斯束：更细的束腰，更快的发散')
axs[0].legend(fontsize=10); axs[0].grid(alpha=.18)
na=np.linspace(.05,.3,300)
axs[1].plot(na,lam/(np.pi*na),color=BLUE,label='束腰半径 w0')
ax2=axs[1].twinx(); ax2.plot(na,2*lam/(np.pi*na**2),color=ORANGE,label='共焦参数 2z_R')
axs[1].set(xlabel='NA_eff（近轴范围）',ylabel='w0 / μm',title='横向尺度 ∝ 1/NA，轴向尺度 ∝ 1/NA²')
ax2.set_ylabel('2z_R / μm',color=ORANGE); axs[1].yaxis.label.set_color(BLUE)
axs[1].grid(alpha=.18)
save(fig,'06-gaussian.png')

# 7. Fiber meridional-ray geometry and mode matching.
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ax=axs[0]
ax.add_patch(Rectangle((0,-1.5),6,3,color='#dce8f4')); ax.add_patch(Rectangle((0,-.8),6,1.6,color='#a4d5ce'))
ax.plot([-2,0,2,6],[1.6,0,-.8,.8],color=ORANGE,lw=2.5)
ax.plot([-2,6],[0,0],color='#6a7585',ls='--',lw=1)
ax.plot([2,2],[-1.45,.35],color='#6a7585',ls=':',lw=1.5)
ax.text(3.3,1.1,'包层 n2',color=BLUE); ax.text(3.3,.4,'纤芯 n1 > n2',color=GREEN)
ax.text(-1.8,-.55,'外界 n0'); ax.text(.8,-.35,'α'); ax.text(1.75,-.42,'i')
ax.text(-1.2,.3,'θa'); ax.text(1.9,-1.32,'法线',ha='right')
ax.set(xlim=(-2.2,6.2),ylim=(-1.7,1.7),title='阶跃型光纤：端面折射 + 侧壁全反射')
ax.axis('off')
r=np.linspace(.2,3,400); overlap=(2*r/(1+r*r))**2
axs[1].plot(r,overlap*100,color=GREEN,lw=2.5); axs[1].axvline(1,color='#8994a5',ls='--')
axs[1].set(xlabel='入射束腰 / 光纤模场半径',ylabel='理想功率耦合效率 / %',title='单模耦合：还要匹配空间模式',ylim=(0,105))
axs[1].grid(alpha=.18)
save(fig,'07-fiber.png')

# 8. D/f approximation in a simple planar aperture geometry.
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
r=np.linspace(.01,1.5,400)
axs[0].plot(r,r/np.sqrt(1+r*r),label='平面孔径几何值',color=BLUE,lw=2.5)
axs[0].plot(r,r,'--',label='近轴近似',color=ORANGE,lw=2)
axs[0].set(xlabel='a / L（孔径半径 / 轴向距离）',ylabel='NA / n',title='只有小角度下，sin θ ≈ tan θ')
axs[0].legend(); axs[0].grid(alpha=.18)
axs[1].plot(r,100*(np.sqrt(1+r*r)-1),color=ORANGE,lw=2.5)
axs[1].set(xlabel='a / L',ylabel='近轴近似的相对高估 / %',title='高 NA 时，直接套 D/(2f) 会偏离几何值')
axs[1].grid(alpha=.18)
save(fig,'08-geometry.png')

# Independent checks: angular integration, Bessel root, pupil integral, mode overlap.
checks=[]
def check(name, calculated, reference, tol=1e-10):
    err=float(abs(calculated-reference)); assert err<tol, (name,err)
    checks.append({'name':name,'calculated':float(calculated),'reference':float(reference),'absolute_error':err})
for n in [1.,1.333,1.515]:
    for q in [.01,.1,.5,.9,.999]:
        theta=np.arcsin(q)
        check(f'collection n={n} NA={n*q}',quad(lambda t:np.sin(t)/2,0,theta)[0],(1-np.sqrt(1-q*q))/2)
for u in [.1,1,2,3.8317059702,5]:
    from scipy.special import j0
    check(f'pupil Fourier integral u={u}',2*quad(lambda r:r*j0(u*r),0,1)[0],2*j1(u)/u)
root=brentq(lambda u: float(j1(u)),3,4)
check('first Airy zero coefficient',root/(2*np.pi),.6098349456332522)
fwhm=brentq(lambda u:float(airy(u))-.5,1,2)/np.pi
check('Airy intensity FWHM coefficient',fwhm,.514496984981094,1e-10)
for w in [.5,1,2]:
    amp=quad(lambda r:2*np.pi*r*np.sqrt(2/np.pi)/w*np.exp(-r*r/w**2)*np.sqrt(2/np.pi)*np.exp(-r*r),0,np.inf)[0]
    check(f'Gaussian overlap w/wf={w}',amp**2,(2*w/(1+w*w))**2)
n1,n2=1.46,1.455; fiber_na=np.sqrt(n1*n1-n2*n2)
check('fiber Snell + TIR',n1*np.sin(np.pi/2-np.arcsin(n2/n1)),fiber_na)
report={'checks_passed':len(checks),'max_absolute_error':max(x['absolute_error'] for x in checks),
 'airy_first_zero_coefficient':root/(2*np.pi),'airy_fwhm_coefficient':fwhm,
 'fiber_example_NA':fiber_na,'fiber_example_half_angle_deg':float(np.degrees(np.arcsin(fiber_na))),
 'fiber_example_V_at_780nm_a2um':float(2*np.pi*2*fiber_na/.780),
 'figures':sorted(p.name for p in OUT.glob('*.png')),'checks':checks}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))


