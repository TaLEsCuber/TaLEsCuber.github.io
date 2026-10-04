"""Reproduce the teaching figures and numerical checks for the spectroscopy article.

Requires Python, NumPy, SciPy and Matplotlib. Run from any working directory.
Figures go to source/img/precision-spectroscopy in the blog checkout; when the
script is downloaded elsewhere they go to a local figures/ directory.
These are idealized model calculations, not experimental data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import voigt_profile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent
OUT = SOURCE / 'img' / 'precision-spectroscopy' if SOURCE.name == 'source' else HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'axes.labelcolor': '#334155',
    'text.color': '#0f172a',
    'axes.edgecolor': '#94a3b8',
    'grid.color': '#e2e8f0',
    'savefig.facecolor': 'white',
})
BLUE, ORANGE, GREEN = '#2563eb', '#d97706', '#0f766e'

def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

# Figure 1: line shapes and their normalized first-order coherence envelopes.
x = np.linspace(-3.5, 3.5, 2001)
sigma = 1 / (2*np.sqrt(2*np.log(2)))
gamma = .5
gaussian = np.exp(-x*x/(2*sigma*sigma))
lorentz = 1/(1+(x/gamma)**2)
voigt = voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
for y, label, color in [(gaussian,'高斯：FWHM = 1',BLUE),
                        (lorentz,'洛伦兹：FWHM = 1',ORANGE),
                        (voigt,'Voigt：两个分量卷积',GREEN)]:
    ax[0].plot(x,y,lw=2.2,label=label,color=color)
ax[0].axhline(.5,color='#94a3b8',ls='--',lw=1)
ax[0].set(xlabel='频率失谐（归一化单位）',ylabel='各自峰值归一化',title='A  单粒子响应与频率分布共同决定线型',ylim=(0,1.07))
ax[0].legend(fontsize=10)
t = np.linspace(0,1.5,500)
ax[1].plot(t,np.exp(-np.pi*t),color=ORANGE,lw=2.2,label='洛伦兹谱 → 指数包络')
ax[1].plot(t,np.exp(-np.pi**2*t*t/(4*np.log(2))),color=BLUE,lw=2.2,label='高斯谱 → 高斯包络')
ax[1].axhline(1/np.e,color='#94a3b8',ls='--',lw=1)
ax[1].set(xlabel=r'时延 × 谱线 FWHM：$\tau\Delta\nu$',ylabel=r'$|g^{(1)}(\tau)|$',title='B  相同 FWHM，不同相干衰减',ylim=(0,1.07))
ax[1].legend(fontsize=10)
for a in ax:a.grid(alpha=.5)
save(fig,'line-shapes.png')

# Figure 2: Morse potential with dimensionless hbar*Omega / De = .18.
q = np.linspace(-.9,4,1000)
eta = .18
fig, ax = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ax[0].plot(q,(1-np.exp(-q))**2,color=BLUE,lw=2.4,label='Morse 势')
ax[0].plot(q,q*q,color=ORANGE,lw=1.8,ls='--',label='平衡位置附近的谐振子')
for v in range(8):
    energy=eta*(v+.5)-eta**2/4*(v+.5)**2
    left=-np.log(1+np.sqrt(energy));right=-np.log(1-np.sqrt(energy))
    ax[0].hlines(energy,left,right,color=GREEN,lw=1.3)
    ax[0].text(right+.07,energy,f'v={v}',fontsize=9,va='center')
ax[0].axhline(1,color='#64748b',ls=':',label='解离极限')
ax[0].set(xlim=(-.9,4),ylim=(0,1.17),xlabel=r'核间距坐标 $a(r-r_e)$',ylabel=r'能量 / $D_e$',title='A  非谐性使高振动态逐渐变密')
ax[0].legend(fontsize=9,loc='upper right')
for j in range(6):
    en=j*(j+1)
    ax[1].hlines(en,0,.9,color=BLUE,lw=2)
    ax[1].text(-.1,en,f'J={j}',ha='right',va='center',fontsize=10)
    if j<5:
        next_en=(j+1)*(j+2)
        xx=1.1+.32*j
        ax[1].annotate('',xy=(xx,next_en),xytext=(xx,en),arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.5))
        ax[1].text(xx+.03,(en+next_en)/2,f'{2*(j+1)}B',fontsize=9,color=GREEN)
ax[1].text(1.05,28,'跃迁波数：2B, 4B, 6B…\n谱线间距为 2B',fontsize=11,va='top',bbox=dict(facecolor='#f0fdfa',edgecolor='none',pad=7))
ax[1].set(xlim=(-.5,3.15),ylim=(-1,33),xticks=[],ylabel=r'转动能量 / $(hcB)$',title='B  刚性转子：能级间距随 J 增大')
ax[1].spines['bottom'].set_visible(False)
save(fig,'molecular-levels.png')

# Figure 3: an exact cosine-transform pair and finite-OPD window transforms.
freq=np.linspace(0,14,3000)
width=.35
centers=[5.,9.];amps=[1.,.65]
spec=sum(a*np.exp(-(freq-f)**2/(2*width**2)) for a,f in zip(amps,centers))
opd=np.linspace(-1.6,1.6,2400)
# Positive Gaussian lines are so far from zero that the negative tail is negligible.
corr=np.sqrt(2*np.pi)*width*np.exp(-2*np.pi**2*width**2*opd**2)*sum(a*np.cos(2*np.pi*f*opd) for a,f in zip(amps,centers))
fig,ax=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
ax[0,0].plot(freq,spec,color=BLUE,lw=2)
ax[0,0].set(xlabel='波数（归一化单位）',ylabel='谱权重',title='A  两条模拟谱线')
ax[0,1].plot(opd,corr,color=GREEN,lw=1.5)
ax[0,1].set(xlabel='光程差（对应倒数单位）',ylabel='扣除直流项的干涉图',title='B  不同条纹周期叠加成干涉图')
wx=np.linspace(-1.3,1.3,1000)
ax[1,0].plot(wx,(np.abs(wx)<1).astype(float),color=ORANGE,lw=2,label='矩形窗')
ax[1,0].plot(wx,np.maximum(1-np.abs(wx),0),color=BLUE,lw=2,label='三角窗')
ax[1,0].set(xlabel='光程差 / 最大光程差 L',ylabel='窗函数',title='C  两种窗口使用相同扫描范围')
ax[1,0].legend()
df=np.linspace(-3,3,2000)
ax[1,1].plot(df,np.sinc(2*df),color=ORANGE,lw=2,label=r'矩形窗 → $\mathrm{sinc}(2L\Delta\tilde\nu)$')
ax[1,1].plot(df,np.sinc(df)**2,color=BLUE,lw=2,label=r'三角窗 → $\mathrm{sinc}^2(L\Delta\tilde\nu)$')
ax[1,1].axhline(0,color='#64748b',lw=.8)
ax[1,1].set(xlabel='波数失谐 × L',ylabel='峰值归一化响应',title='D  压低旁瓣的代价是加宽主峰')
ax[1,1].legend(fontsize=9)
for a in ax.flat:a.grid(alpha=.5)
save(fig,'fourier-spectroscopy.png')

# Independent calculations: quadrature and root finding, plus the examples.
fwhm_rect=2*brentq(lambda z:np.sinc(2*z)-.5,.001,.49)
fwhm_tri=2*brentq(lambda z:np.sinc(z)**2-.5,.001,.99)
quad_errors=[]
for delay in [0,.013,.11,.37,.9]:
    numerical=quad(lambda f:sum(a*np.exp(-(f-c)**2/(2*width**2)) for a,c in zip(amps,centers))*np.cos(2*np.pi*f*delay),0,14,epsabs=1e-10)[0]
    analytic=np.sqrt(2*np.pi)*width*np.exp(-2*np.pi**2*width**2*delay**2)*sum(a*np.cos(2*np.pi*c*delay) for a,c in zip(amps,centers))
    quad_errors.append(abs(numerical-analytic))
normalized_lorentz=lambda z:gamma/(np.pi*(z*z+gamma*gamma))
normalized_gauss=lambda z:np.exp(-z*z/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi))
voigt_errors=[]
for detuning in [0,.3,1.5,4]:
    convolution=quad(lambda z:normalized_gauss(z)*normalized_lorentz(detuning-z),-np.inf,np.inf,epsabs=1e-11)[0]
    voigt_errors.append(abs(convolution-voigt_profile(detuning,sigma,gamma)))
nu_l=const.c/(532e-9);nu_v=const.c*1000*100
report={
    'one_inverse_cm_GHz':const.c*100/1e9,
    'one_inverse_cm_meV':const.h*const.c*100/const.e*1000,
    'natural_FWHM_for_10ns_MHz':1/(2*np.pi*10e-9)/1e6,
    'doppler_780nm_87u_300K_MHz':1/(780e-9)*np.sqrt(8*const.k*300*np.log(2)/(87*const.atomic_mass))/1e6,
    'stokes_nm_532nm_1000cm_inverse':1/(1/532-1000/1e7),
    'antiStokes_Stokes_power_ratio':((nu_l+nu_v)/(nu_l-nu_v))**4*np.exp(-const.h*nu_v/(const.k*300)),
    'rect_window_FWHM_times_L':fwhm_rect,
    'triangle_window_FWHM_times_L':fwhm_tri,
    'opd_step_for_4000_inverse_cm_um':1/(2*4000*100)*1e6,
    'dual_comb_delay_step_fs':100/(100e6)**2*1e15,
    'dual_comb_scan_ms':1/100*1000,
    'cosine_transform_max_absolute_error':max(quad_errors),
    'voigt_convolution_max_absolute_error':max(voigt_errors),
}
assert max(quad_errors)<1e-9
assert max(voigt_errors)<1e-9
assert .602<fwhm_rect<.604 and .885<fwhm_tri<.887
(HERE/'numerical-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print(f'Figures saved in {OUT}')
