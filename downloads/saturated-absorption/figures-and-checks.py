"""Original teaching figures; no curves are laboratory data.
Run: python figures-and-checks.py --output path/to/img/saturated-absorption
Dependencies: numpy, matplotlib. Chinese font: Microsoft YaHei or Noto Sans CJK SC.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'img' / 'saturated-absorption')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'],
    'font.size':13, 'axes.titlesize':16, 'axes.labelsize':13, 'axes.unicode_minus':False,
    'figure.facecolor':'#f6f8fb', 'axes.facecolor':'#ffffff', 'axes.spines.top':False,
    'axes.spines.right':False, 'text.color':'#18334b', 'axes.labelcolor':'#18334b', 'xtick.color':'#496173', 'ytick.color':'#496173'})
BLUE='#2778b8'; ORANGE='#df8438'; GREEN='#239979'; NAVY='#18334b'; PURPLE='#8056b5'; GREY='#8093a1'

def finish(fig,name,title,subtitle):
    fig.suptitle(title, fontsize=23, fontweight='bold', x=.06, ha='left', y=.97)
    fig.text(.06,.905,subtitle,fontsize=12,color='#5c7282')
    fig.savefig(args.output/name,dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)

def arrow(ax,a,b,color=NAVY,style='-',lw=2.4):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':lw,'color':color,'linestyle':style})

def box(ax,x,y,w,h,label,color=NAVY,fs=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014,rounding_size=0.025',facecolor='white',edgecolor=color,lw=1.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=color)

# 1: selected velocity classes. u = vz / sigma_v, d = lambda*detuning/sigma_v.
fig,axs=plt.subplots(1,2,figsize=(14,6.8)); fig.subplots_adjust(top=.79,bottom=.2,wspace=.25,left=.07,right=.96)
v=np.linspace(-3.5,3.5,1600); f=np.exp(-v*v/2)
for ax,d,title in zip(axs,[1.25,0],['偏离共振：两束光选中不同速度群',r'在线中心：共同作用于 $v_z \simeq 0$']):
    ax.plot(v,f,color=GREY,lw=2,label='热运动速度分布')
    for center,c,lab in [(d,ORANGE,'泵浦：沿 +z'),(-d,BLUE,'探测：沿 −z')]:
        selected=np.exp(-((v-center)/.17)**2)*f
        ax.fill_between(v,0,selected,color=c,alpha=.4,label=lab)
        ax.axvline(center,color=c,lw=1.8,ls='--')
    ax.set(xlim=(-3.5,3.5),ylim=(0,1.3),xlabel=r'纵向速度 $v_z/\sigma_v$',ylabel='相对布居 / 被选中的速度群',title=title)
    ax.legend(loc='upper left',fontsize=10)
fig.text(.5,.065,'同频、反向、空间重叠   →   同一窄速度群的交叉饱和；横向热运动仍然存在',ha='center',fontsize=15)
finish(fig,'01-velocity-selection.png','01  两束反向光，如何从热气体中选出窄谱线','教学模型 · 色带宽度仅表示有限共振范围，不是论文的实际速度分布')

# 2: explicitly phenomenological OD model, all positive.
fig,axs=plt.subplots(1,3,figsize=(15,6.6)); fig.subplots_adjust(top=.78,bottom=.19,wspace=.28,left=.065,right=.97)
s=np.linspace(0,12,500); axs[0].plot(s,s/(2*(1+s)),color=ORANGE,lw=3)
axs[0].axhline(.5,color=GREY,ls='--'); axs[0].text(5,.44,'上限 1/2',fontsize=12)
axs[0].set(xlabel=r'共振饱和参数 $s_0=I/I_{sat}$',ylabel=r'激发态布居 $\rho_{ee}$',title='饱和 ≠ 全部原子都被激发',ylim=(0,.57))
x=np.linspace(-8,8,1601); broad=1.5*np.exp(-x*x/32); od=broad-.27/(1+4*x*x)
axs[1].plot(x,broad,'--',color=GREY,label='泵浦关闭'); axs[1].plot(x,od,color=ORANGE,lw=2.5,label='泵浦打开')
axs[1].set(xlabel=r'失谐 $\delta/\gamma$',ylabel='光学厚度 OD（模型）',title='吸收中：窄凹陷'); axs[1].legend(fontsize=10)
axs[2].plot(x,np.exp(-broad),'--',color=GREY); axs[2].plot(x,np.exp(-od),color=BLUE,lw=2.5)
axs[2].set(xlabel=r'失谐 $\delta/\gamma$',ylabel='透射率 T（模型）',title='透射中：窄凸峰')
fig.text(.5,.065,'右侧两图使用同一模型：T = exp(−OD)。纵轴不同，“峰”和“谷”的方向也不同。',ha='center',fontsize=13)
finish(fig,'02-saturation-and-spectra.png','02  饱和、吸收凹陷与透射凸峰','理想闭合二能级布居 + 示意光学厚度；不用于拟合实际碘分子多能级光谱')

# 3: slow frequency dither model, not the paper's 10 MHz FM line shape.
fig,axs=plt.subplots(1,2,figsize=(14,6.6)); fig.subplots_adjust(top=.78,bottom=.2,wspace=.28,left=.07,right=.96)
x=np.linspace(-2,2,801); a=.12; theta=np.linspace(0,2*np.pi,4096,endpoint=False)
T=lambda q: .3+.2/(1+4*q*q)
dT=lambda q: -1.6*q/(1+4*q*q)**2
e=2*np.mean(T(x[:,None]+a*np.cos(theta))*np.cos(theta),axis=1)
axs[0].plot(x,T(x),lw=3,color=BLUE); axs[0].axvline(0,color=GREY,ls='--'); axs[0].set(xlabel=r'失谐 $\delta/\gamma$',ylabel='局部透射率',title='峰顶本身不告诉我们失谐方向')
axs[1].plot(x,e,color=ORANGE,lw=3,label='数值同步解调'); axs[1].plot(x,a*dT(x),'--',color=NAVY,lw=1.6,label='小调制近似 a·dT/dx')
axs[1].axhline(0,color=GREY,lw=1); axs[1].axvline(0,color=GREY,ls='--'); axs[1].legend(fontsize=11)
axs[1].set(xlabel=r'失谐 $\delta/\gamma$',ylabel='解调信号（任意单位）',title='过零信号给出方向与大小')
fig.text(.5,.067,'慢频率抖动示例：a = 0.12γ；改变解调相位或探测极性，会把误差信号整体翻转。',ha='center',fontsize=13)
finish(fig,'03-error-signal.png','03  从对称谱线到有正负号的误差信号','教学模型 · 低频、准静态、小幅调制近似；不能代替论文 10 MHz 相位调制的完整模型')

# 4: topology faithful to supplied Fig 1: common EOM before pump/probe split.
fig,ax=plt.subplots(figsize=(15,8.8)); fig.subplots_adjust(top=.85,bottom=.05,left=.035,right=.97); ax.set(xlim=(0,15),ylim=(0,8)); ax.axis('off')
box(ax,.2,5.3,2.0,1.0,'1108 nm\n种子激光',NAVY)
box(ax,3,5.3,1.8,1.0,'倍频 / 和频\n光源模块',NAVY); arrow(ax,(2.2,5.8),(3,5.8))
box(ax,6,6.4,2.6,.9,'369 nm → 离子阱',PURPLE); arrow(ax,(4.8,6),(5.5,6),PURPLE); arrow(ax,(5.5,6),(5.5,6.85),PURPLE); arrow(ax,(5.5,6.85),(6,6.85),PURPLE)
box(ax,6,4.8,1.5,.9,'EOM\n10 MHz',GREEN); arrow(ax,(4.8,5.5),(5.5,5.5),GREEN); arrow(ax,(5.5,5.5),(5.5,5.25),GREEN); arrow(ax,(5.5,5.25),(6,5.25),GREEN)
ax.text(4.9,4.75,'554 nm',color=GREEN,fontsize=12)
box(ax,8.1,4.8,1.5,.9,'HWP / PBS\n分光',GREEN); arrow(ax,(7.5,5.25),(8.1,5.25),GREEN)
box(ax,10.5,4.55,1.8,1.25,'碘气室\n10 cm',GREEN)
arrow(ax,(9.6,5.05),(10.5,5.05),BLUE); arrow(ax,(12.3,5.05),(13.1,5.05),BLUE)
box(ax,13.15,4.7,1.3,.7,'PD',BLUE)
arrow(ax,(8.85,5.7),(8.85,6.4),ORANGE); box(ax,9.3,6.05,2.7,.75,'泵浦缩束：23 mW',ORANGE,12)
arrow(ax,(8.85,6.4),(9.3,6.4),ORANGE); arrow(ax,(12,6.4),(12.75,6.4),ORANGE); arrow(ax,(12.75,6.4),(12.75,5.55),ORANGE); arrow(ax,(12.75,5.55),(10.5,5.55),ORANGE)
ax.text(9.3,4.3,'探测：3 mW →',color=BLUE,fontsize=12)
box(ax,10.5,2.8,3.0,.8,'解调 + 低通 + PID',NAVY); arrow(ax,(13.8,4.7),(13.8,3.2)); arrow(ax,(13.8,3.2),(13.5,3.2))
box(ax,6,2.8,2.6,.8,'10 MHz 本振 / 移相',GREY,12); arrow(ax,(7.0,3.6),(7.0,4.8),GREY,'--'); arrow(ax,(8.6,3.2),(10.5,3.2),GREY,'--')
arrow(ax,(12,2.8),(12,2.25)); arrow(ax,(12,2.25),(1.2,2.25)); arrow(ax,(1.2,2.25),(1.2,5.3))
ax.text(3.0,2.43,'反馈到种子激光 PZT（论文报告的执行器）',fontsize=12)
box(ax,.3,.45,5.4,.85,'1108 nm 分出少量光 + 频率梳 → 拍频计数',GREY,12)
arrow(ax,(.6,5.3),(.6,1.3),GREY,'--'); ax.text(6.15,.86,'环外频率监测；与上方稳频反馈区分',color=GREY,fontsize=12)
finish(fig,'04-lab-frequency-chain.png','04  用绿光中的碘分子参考，稳定紫外冷却光','依据两篇参考文献重绘的功能关系图 · 省略折叠镜和部分偏振元件，不表示实际台面坐标')

# 5: true afocal Kepler ray geometry f1=3, f2=1; coordinate reversal after focus.
fig,axs=plt.subplots(1,2,figsize=(14,6.8),gridspec_kw={'width_ratios':[1.2,1]}); fig.subplots_adjust(top=.78,bottom=.17,wspace=.22,left=.05,right=.96)
ax=axs[0]; ax.set(xlim=(-1.2,5.4),ylim=(-1.65,1.65)); ax.axis('off')
ax.axhline(0,color=GREY,ls='--',lw=1)
for h in [-.95,.95]: ax.plot([-1,0,3,4,5.2],[h,h,0,-h/3,-h/3],color=GREEN,lw=2.7)
for xx,hh in [(0,2.5),(4,1.5)]: ax.add_patch(Ellipse((xx,0),.15,hh,facecolor='#aedcea',edgecolor=BLUE,lw=1.5))
ax.scatter([3],[0],color=ORANGE,s=30); ax.text(3,-.38,'共同焦点',ha='center',fontsize=12)
ax.text(0,1.45,r'先遇到 $L_1$',ha='center'); ax.text(4,1.05,r'再遇到 $L_2$',ha='center')
ax.text(1.5,-1.18,r'$f_1$',ha='center',fontsize=17); ax.text(3.5,-1.18,r'$f_2$',ha='center',fontsize=17)
arrow(ax,(0,-.97),(3,-.97),GREY); arrow(ax,(3,-.97),(4,-.97),GREY)
ax.text(1.8,1.45,r'$f_1>f_2>0$',ha='center',fontsize=16)
ax.text(2,-1.58,'几何示例：半径缩为 1/3，强度约增至 9 倍',ha='center',fontsize=12)
r=np.linspace(.15,1,250); axs[1].plot(r,1/r**2,color=ORANGE,lw=3)
axs[1].scatter([1/3],[9],color=NAVY,s=55,zorder=4); axs[1].annotate('M = 1/3 → 9 倍',xy=(1/3,9),xytext=(.43,19),arrowprops={'arrowstyle':'->','color':NAVY},fontsize=12)
axs[1].set(xlabel=r'半径比 $M=w_{out}/w_{in}=f_2/f_1$',ylabel=r'强度比 $I_{out}/I_{in}$（无损）',ylim=(0,47),title='强度按面积的倒数增加')
finish(fig,'05-kepler-compression.png','05  缩小光斑，提高局部泵浦强度',r'理想薄透镜、近轴、无损模型 · 两镜间距 $f_1 + f_2$；实际光束应再检查准直与重叠')

# 6: explicit toy optimization, not a fit to reported iodine numbers.
fig,axs=plt.subplots(1,2,figsize=(14,6.7)); fig.subplots_adjust(top=.78,bottom=.19,wspace=.26,left=.075,right=.96)
s=np.linspace(0,8,800); A=s/(1+s); g=np.sqrt(1+s); k=A/g**2
axs[0].plot(s,A,label='对比度 A/A∞',lw=3,color=GREEN); axs[0].plot(s,g/g[-1],label='线宽 γ / γ(s=8)',lw=3,color=ORANGE)
axs[0].set(xlabel='饱和参数 s（教学模型）',ylabel='分别归一化的量',title='信号变强，也可能同时变宽'); axs[0].legend(fontsize=11)
axs[1].plot(s,k/.25,color=BLUE,lw=3); axs[1].axvline(1,color=GREY,ls='--'); axs[1].text(1.2,.93,'本模型最优 s = 1',fontsize=12)
axs[1].set(xlabel='饱和参数 s（教学模型）',ylabel='归一化零点斜率 |K|',title=r'固定小调制幅度：$|K|\propto A/\gamma^2$',ylim=(0,1.15))
fig.text(.5,.06,'模型假设：A ∝ s/(1+s)，γ ∝ √(1+s)，电子噪声不变。实际最佳点必须由测量决定。',ha='center',fontsize=12)
finish(fig,'06-slope-tradeoff.png','06  最值得优化的是鉴频能力，而不仅是峰峰值','模型说明权衡机制；不拟合、不复现论文中的功率密度扫描数据')

# Independent numerical sanity checks of derivations.
f1,f2=150.,50.
P=lambda d: np.array([[1.,d],[0.,1.]])
L=lambda f: np.array([[1.,0.],[-1/f,1.]])
matrix=L(f2)@P(f1+f2)@L(f1)
assert abs(matrix[1,0])<1e-14 and abs(matrix[0,0]+f2/f1)<1e-14
xt=np.linspace(-1.9,1.9,27); tiny=1e-4
enum=2*np.mean(T(xt[:,None]+tiny*np.cos(theta))*np.cos(theta),axis=1)/tiny
derivative_error=float(np.max(np.abs(enum-dT(xt))))
assert derivative_error<1e-7
assert abs(2*np.mean(T(tiny*np.cos(theta))*np.cos(theta)))<1e-14
lam=554e-9; delta=12e6; v=lam*delta
assert abs(delta-v/lam)<1e-7 and abs(delta+(-v)/lam)<1e-7
nu1,nu2=100e6,200e6; nuc=(nu1+nu2)/2; vc=lam*(nu2-nu1)/2
assert abs(nuc-vc/lam-nu1)<1e-7 and abs(nuc+vc/lam-nu2)<1e-7
thermal_fwhm=2*math.sqrt(2*math.log(2))*math.sqrt(1.380649e-23*300/(254*1.66053906660e-27))/lam
report={'all_checks_passed':True,'figure_count':6,'curves':'teaching models, not measured laboratory data',
    'lockin_derivative_max_abs_error':derivative_error,'kepler_abcd':matrix.tolist(),
    'iodine_300K_554nm_doppler_fwhm_MHz':thermal_fwhm/1e6,
    'gaussian_knife_10_90_width_over_w':1.2815515655446004,
    'paper_intensity_291_42_mW_mm2_as_W_cm2':29.142,
    'fractional_4_4e_11_at_369nm_equivalent_Hz':4.4e-11*299792458/(369e-9),
    'note':'Equivalent Hz is averaged frequency fluctuation scale, NOT optical linewidth.'}
(Path(__file__).parent/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
