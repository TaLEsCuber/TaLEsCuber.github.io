"""Original PDH tutorial models, figures and consistency checks.

Run: python source/downloads/pdh/figures-and-checks.py
Dependencies: numpy, scipy, matplotlib. No experimental data is used.
Optical convention: exp(+i omega t); propagation exp(-i k L).
Demodulation: ideal multiplication by 2 sin(Omega t), then low-pass.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import jv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'img' / 'pdh'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','DejaVu Sans'], 'axes.unicode_minus':False, 'font.size':11, 'figure.facecolor':'#faf8f2', 'axes.facecolor':'#faf8f2', 'axes.spines.top':False, 'axes.spines.right':False, 'text.color':'#20334b', 'axes.labelcolor':'#20334b', 'svg.fonttype':'none'})
BLUE, RED, TEAL, GOLD = '#3565a3', '#c34d48', '#087f8c', '#bd8424'
C, LENGTH, FINESSE, FM, BETA = 299792458., .1, 100000., 20e6, .3
FSR = C/(2*LENGTH)
WIDTH = FSR/FINESSE
s = np.sin(np.pi/(2*FINESSE))
R = (np.sqrt(1+s*s)-s)**2
r = np.sqrt(R)

def reflection(d):
    # expm1 avoids subtractive cancellation around resonance.
    zminus1 = np.expm1(-2j*np.pi*np.asarray(d)/FSR)
    return -r*zminus1 / ((1-R)-R*zminus1)

def error(d, beta=BETA, order=8):
    """Voltage divided by GP, full sum of adjacent sideband beats."""
    z = np.zeros_like(np.asarray(d), dtype=complex)
    for n in range(-order, order):
        z += jv(n+1,beta)*jv(n,beta)*reflection(np.asarray(d)+(n+1)*FM)*np.conj(reflection(np.asarray(d)+n*FM))
    return -2*z.imag

def save(fig, name):
    for ext in ['png','svg']:
        fig.savefig(OUT/(name+'.'+ext), dpi=175, bbox_inches='tight')
    plt.close(fig)

# 1. Functional optical and electrical diagram. No PBS polarization geometry implied.
fig, ax = plt.subplots(figsize=(13,6.2))
ax.set(xlim=(0,13),ylim=(0,6.5)); ax.axis('off')
def box(x,y,w,h,label,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=.12',edgecolor=color,facecolor='white',lw=1.6))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
def arrow(points,color=RED,ls='-'):
    for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2,ls=ls)
    ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops={'arrowstyle':'->','color':color,'lw':2,'linestyle':ls})
box(.25,4.9,1.15,.8,'激光器'); box(1.9,4.9,1.1,.8,'隔离器')
box(3.5,4.9,1.2,.8,'相位 EOM'); box(5.3,4.9,1.6,.8,'模式匹配\n透镜')
box(7.5,4.9,1.4,.8,'反射光\n分离组件'); box(9.5,4.9,1.4,.8,'参考腔')
box(11.5,4.9,1.2,.8,'透射 PD')
for a,b in [(1.4,1.9),(3,3.5),(4.7,5.3),(6.9,7.5),(8.9,9.5),(10.9,11.5)]: arrow([(a,5.3),(b,5.3)])
arrow([(9.5,5.02),(8.9,5.02)],RED)
arrow([(8.2,4.9),(8.2,3.85)],RED)
ax.text(9,4.2,'腔反射',color=RED,fontsize=10)
box(7.5,3.05,1.4,.8,'反射 RF-PD')
box(7.5,1.4,1.4,.8,'混频器')
arrow([(8.2,3.05),(8.2,2.2)],TEAL)
ax.text(8.45,2.6,'RF',color=TEAL)
box(3.5,3.05,1.2,.8,'RF 源')
arrow([(4.1,3.85),(4.1,4.9)],TEAL)
box(5.3,1.4,1.2,.8,'移相器')
arrow([(4.1,3.05),(4.1,1.8),(5.3,1.8)],TEAL)
arrow([(6.5,1.8),(7.5,1.8)],TEAL); ax.text(6.85,2,'LO',color=TEAL)
box(9.6,1.4,1.15,.8,'低通')
arrow([(8.9,1.8),(9.6,1.8)],TEAL)
box(10.1,.1,2.5,.7,'伺服：快路 + 慢路')
arrow([(10.75,1.8),(11.35,1.8),(11.35,.8)],TEAL)
arrow([(10.1,.45),(.8,.45),(.8,4.9)],BLUE,'--')
ax.text(2.1,.7,'快路 → 电流 / 执行 EOM；慢路 → PZT / 温控',color=BLUE,fontsize=10)
ax.text(.25,6.1,'PDH 的闭环：相位调制 → 反射探测 → 同步解调 → 频率反馈',fontsize=16,weight='bold')
ax.text(.25,-.2,'红色：光路（功能示意）   绿色：RF / 解调电信号   蓝色虚线：反馈；透射 PD 用于辅助判锁',fontsize=10)
save(fig,'01-system')

# 2. Optical spectrum, cavity response, and complex reflection locus.
fig, axes = plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
n=np.arange(-3,4); powers=jv(n,BETA)**2
axes[0].vlines(n,0,powers,color=TEAL,lw=4); axes[0].scatter(n,powers,color=TEAL)
axes[0].set(xlabel='相对载波频移 / fm',ylabel='功率 / 总入射功率',title='相位调制：β = 0.30',ylim=(0,1.12))
axes[0].set_yticks(np.arange(0,1.01,.2))
for ni in [-1,0,1]: axes[0].annotate(f'{100*jv(ni,BETA)**2:.2f}%',(ni,jv(ni,BETA)**2),xytext=(0,9),textcoords='offset points',ha='center')
x=np.linspace(-5,5,1801); rc=reflection(x*WIDTH/2)
axes[1].plot(x,abs(rc)**2,label='反射功率',color=BLUE)
axes[1].plot(x,rc.imag,label='反射场虚部',color=RED)
axes[1].axhline(0,color='gray',lw=.6); axes[1].legend()
axes[1].set(xlabel='失谐 x = 2Δν / δνc',title='功率偶对称，虚部奇对称')
axes[2].plot(rc.real,rc.imag,color=TEAL)
for xi in [-1,0,1]:
    ri=reflection(xi*WIDTH/2); axes[2].scatter(ri.real,ri.imag,color=RED); axes[2].annotate(f'x = {xi}',(ri.real,ri.imag),xytext=(8,5),textcoords='offset points')
axes[2].set(xlabel='Re(r)',ylabel='Im(r)',title='反射场复平面轨迹',xlim=(-.15,1.3),ylim=(-.65,.65)); axes[2].set_aspect('equal')
save(fig,'02-sidebands-reflection')

# 3. Full response and near-resonance linearity.
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
d=np.unique(np.concatenate([np.linspace(-30e6,30e6,6001)]+[np.linspace(center-10*WIDTH,center+10*WIDTH,2001) for center in [-FM,0,FM]]))
axes[0].plot(d/1e6,error(d),color=TEAL)
axes[0].set(xlabel='载波失谐 Δν / MHz',ylabel='解调电压 / ($GP_0$)',title='大范围扫描：主共振与边带共振')
for pos,label in [(-20,'上边带入腔'),(0,'载波入腔'),(20,'下边带入腔')]:
    axes[0].axvline(pos,color=GOLD,ls=':',lw=1); axes[0].text(pos,.21,label,ha='center',fontsize=10)
d=np.linspace(-2*WIDTH,2*WIDTH,2001); slope=8*jv(0,BETA)*jv(1,BETA)/WIDTH
axes[1].plot(d/WIDTH,error(d),label='完整边带求和',color=TEAL,lw=2.5)
axes[1].plot(d/WIDTH,slope*d,label='零点切线',color=GOLD,ls='--')
axes[1].axvspan(-.1,.1,color=TEAL,alpha=.12,label='±0.1 倍腔线宽')
axes[1].set(xlabel='Δν / 腔功率 FWHM',ylabel='解调电压 / ($GP_0$)',title='误差信号的中央色散形状',ylim=(-.4,.4)); axes[1].legend(fontsize=10,loc='lower right')
save(fig,'03-error-signal')

# 4. Demodulation phase and DC offset.
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
x=np.linspace(-2,2,1001); base=x/(1+x*x)
for phase,col in [(0,TEAL),(45,BLUE),(90,GOLD),(180,RED)]: axes[0].plot(x,base*np.cos(np.deg2rad(phase)),label=f'相位偏差 {phase}°',color=col)
axes[0].legend(fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2); axes[0].set(xlabel='x = 2Δν / δνc',ylabel='归一化误差信号',title='理想解调：转相位会改变斜率')
for offset,col in [(0,TEAL),(.12,RED)]: axes[1].plot(x,base+offset,label=f'归一化偏置 {offset:.2f}',color=col)
axes[1].axhline(0,color='gray',lw=1); axes[1].axvline(0,color='gray',ls=':'); axes[1].legend()
axes[1].set(xlabel='x = 2Δν / δνc',title='RAM / 电偏置会移动锁定零点')
save(fig,'04-phase-offset')

# 5. Explicit toy open loop with cavity pole and compensation.
f=np.logspace(1,7,6000)
def loop(f,ug):
    def raw(v):
        return (1+1j*v/(WIDTH/2))/(1j*v*(1+1j*v/200e3)*(1+1j*v/(WIDTH/2))*(1+1j*v/500e3))*np.exp(-2j*np.pi*v*200e-9)
    return raw(f)/abs(raw(ug))
fig,axes=plt.subplots(3,1,figsize=(10.5,9),sharex=True,layout='constrained')
margins={}
for ug,col in [(80e3,TEAL),(180e3,RED)]:
    ol=loop(f,ug); pm=180+np.angle(loop(ug,ug),deg=True); margins[str(int(ug))]=float(pm)
    label=f'单位增益 {ug/1e3:.0f} kHz；相位裕度 {pm:.1f}°'
    axes[0].semilogx(f,20*np.log10(abs(ol)),color=col,label=label)
    axes[1].semilogx(f,np.unwrap(np.angle(ol))*180/np.pi,color=col)
    axes[2].semilogx(f,20*np.log10(abs(1/(1+ol))),color=col)
    for ax in axes: ax.axvline(ug,color=col,ls=':',alpha=.6)
axes[0].axhline(0,color='gray',ls='--'); axes[0].legend(); axes[0].set(ylabel='开环幅度 / dB',ylim=(-65,90),title='同一模型提高增益：带宽增加，裕度减小')
axes[1].axhline(-180,color='gray',ls='--'); axes[1].set(ylabel='开环相位 / °',ylim=(-260,-70))
axes[2].axhline(0,color='gray',ls='--'); axes[2].set(ylabel='自由激光噪声传递 |S| / dB',xlabel='傅里叶频率 f / Hz',ylim=(-80,22),xlim=(10,2e6))
for ax in axes: ax.grid(alpha=.15)
save(fig,'05-servo')

# Independent checks: energy, cavity width, demodulation from time-domain field,
# central derivative vs analytic expression, full orders vs three frequencies.
checks={}
checks['bessel_power_sum']=float(np.sum(jv(np.arange(-12,13),BETA)**2))
assert abs(checks['bessel_power_sum']-1)<1e-13
checks['half_width_reflected_power']=float(abs(reflection(WIDTH/2))**2)
assert abs(checks['half_width_reflected_power']-.5)<1e-9
checks['odd_symmetry_error']=float(np.max(abs(error(d)+error(-d))))
assert checks['odd_symmetry_error']<1e-12
checks['slope_numeric_per_Hz']=float((error(1.)-error(-1.))/2)
checks['slope_approx_per_Hz']=float(slope)
assert abs(checks['slope_numeric_per_Hz']/slope-1)<2e-6
theta=np.linspace(0,2*np.pi,32768,endpoint=False)
test_d=WIDTH*.2
field=sum(jv(n,BETA)*reflection(test_d+n*FM)*np.exp(1j*n*theta) for n in range(-8,9))
checks['direct_time_demod']=float(np.mean(abs(field)**2*2*np.sin(theta)))
checks['frequency_sum_demod']=float(error(test_d))
assert abs(checks['direct_time_demod']-checks['frequency_sum_demod'])<1e-12
three=-2*jv(0,BETA)*jv(1,BETA)*np.imag(reflection(test_d+FM)*np.conj(reflection(test_d))-reflection(test_d)*np.conj(reflection(test_d-FM)))
checks['three_frequency_relative_error']=float(abs(three/error(test_d)-1))
checks['linear_approx_relative_error_at_0p1_FWHM']=float(abs(error(.1*WIDTH)/(slope*.1*WIDTH)-1))
P,G=.2e-3,1e4
checks.update({'FSR_Hz':FSR,'cavity_FWHM_Hz':WIDTH,'cavity_pole_Hz':WIDTH/2,'power_lifetime_s':1/(2*np.pi*WIDTH),'mirror_power_reflectivity':float(R),'carrier_power_fraction':float(jv(0,BETA)**2),'each_first_sideband_power_fraction':float(jv(1,BETA)**2),'example_slope_V_per_Hz':float(G*P*slope),'example_1mV_offset_Hz':float(.001/(G*P*slope)),'loop_phase_margins_deg':margins,'status':'all checks passed','data_type':'illustrative model, not measurement'})
(HERE/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
