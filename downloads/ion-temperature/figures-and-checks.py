"""Reproduce original SVG teaching figures with Python 3 (standard library only).

Run: python source/downloads/ion-temperature/figures-and-checks.py
All curves are model calculations or explicitly marked cartoons, not lab data.
No external fonts, network, plotting packages, or experimental inputs required.
"""
from pathlib import Path
from math import exp, log, log1p, sqrt, sin, cos, pi, factorial
from html import escape
import json

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent
OUT = SOURCE / 'img' / 'ion-temperature' if SOURCE.name == 'source' else HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
K = 1.380649e-23
H = 6.62607015e-34
HB = H / (2*pi)
U = 1.66053906892e-27
M = 171*U  # nominal mass; teaching example, not a precision isotope mass
F = 1e6
W = 2*pi*F
LAM = 369.5e-9
BLUE, RED, TEAL, GOLD, INK, GRAY = '#2563eb', '#dc5b46', '#0f807d', '#a66a10', '#172b43', '#64748b'

def grid(lo, hi, n=501):
    return [lo+(hi-lo)*i/(n-1) for i in range(n)]

def trap(values, dx):
    return dx*(sum(values)-(values[0]+values[-1])/2)

def thermal(nbar, n):
    if nbar == 0:
        return 1.0 if n == 0 else 0.0
    return exp(n*log(nbar/(nbar+1))-log1p(nbar))

def ncut(nbar, tol=1e-13):
    return 1 if nbar == 0 else max(2, int(log(tol)/log(nbar/(1+nbar)))+2)

def laguerre(n, alpha, x):
    a, b = 1., 1+alpha-x
    if n == 0:
        return a
    for j in range(1, n):
        a, b = b, ((2*j+1+alpha-x)*b-(j+alpha)*a)/(j+1)
    return b

class SVG:
    def __init__(self, title, subtitle, height=620):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img"><title>{escape(title)}</title>',
                      '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#172b43} .small{font-size:16px}</style>',
                      f'<rect width="1100" height="{height}" rx="20" fill="#f6f8fc"/>']
        self.text(42, 48, title, 29, weight='bold')
        self.text(42, 82, subtitle, 17, GRAY)
    def text(self, x, y, txt, size=19, color=INK, anchor='start', weight='normal'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(txt)}</text>')
    def rect(self, x,y,w,h,fill='#ffffff',stroke='#dbe3ef',rx=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x1,y1,x2,y2,color=GRAY,width=2,dash=''):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def path(self, points, color=BLUE, width=3, fill='none'):
        p=' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
        self.parts.append(f'<polyline points="{p}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
    def circle(self,x,y,r=5,color=BLUE):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')
    def note(self, txt):
        self.text(42, self.height-24, txt, 15, GRAY)
    def save(self, name):
        (OUT/name).write_text('\n'.join(self.parts+['</svg>']),encoding='utf-8')

class Axis:
    def __init__(self,s,x,y,w,h,xlim,ylim,xlabel,ylabel,xticks,yticks):
        self.s,self.x,self.y,self.w,self.h=s,x,y,w,h
        self.xlim,self.ylim=xlim,ylim
        s.rect(x-1,y-1,w+2,h+2,rx=0)
        for t in xticks:
            xx,yy=self.xy(t,ylim[0]); s.line(xx,y,xx,y+h,'#e3e9f2',1)
            s.text(xx,y+h+26,f'{t:g}',15,GRAY,'middle')
        for t in yticks:
            xx,yy=self.xy(xlim[0],t); s.line(x,yy,x+w,yy,'#e3e9f2',1)
            s.text(x-10,yy+5,f'{t:g}',15,GRAY,'end')
        s.line(x,y+h,x+w,y+h,INK,1.5);s.line(x,y,x,y+h,INK,1.5)
        s.text(x+w/2,y+h+58,xlabel,18,anchor='middle')
        s.text(x,y-18,ylabel,18)
    def xy(self,a,b):
        return (self.x+self.w*(a-self.xlim[0])/(self.xlim[1]-self.xlim[0]),
                self.y+self.h*(1-(b-self.ylim[0])/(self.ylim[1]-self.ylim[0])))
    def plot(self,xs,ys,color=BLUE,width=3):
        self.s.path([self.xy(x,y) for x,y in zip(xs,ys)],color,width)

# 1. Overview: same motion seen through position, velocity, and phonon number.
s=SVG('离子温度：把看不见的运动变成可测的信号','三条测量入口：位置分布、速度分布、声子布居。先确定运动模，再选择温度计。')
headers=['看位置','看速度','看声子数']
descs=['成像测温','荧光谱线 / 再冷却 / 暗共振','边带不对称 / Rabi 振荡']
for i in range(3):
    x=38+i*354;s.rect(x,118,330,354);s.text(x+24,160,headers[i],24,weight='bold')
    s.text(x+24,440,descs[i],17,TEAL)
xv=grid(-3,3,181)
for i,color in [(0,BLUE),(1,TEAL)]:
    x=66+i*354
    s.line(x,360,x+270,360,GRAY,1.5)
    s.path([(x+(v+3)*45,360-125*exp(-v*v/2)) for v in xv],color)
    s.text(x+132,398,'位置 x' if i==0 else '速度 v∥',18,anchor='middle')
for n in range(7):
    x=782+n*33;hh=130*thermal(2,n)/thermal(2,0)
    s.rect(x,360-hh,21,hh,BLUE,BLUE,2);s.text(x+10,385,str(n),15,anchor='middle')
s.text(890,408,'声子数 n',18,anchor='middle')
s.rect(38,494,1038,64,'#e6f1f1','#c5e2df')
s.text(557,534,'输出：某一运动模的 n̄ 或 T  ＋  不确定度  ＋  模型适用条件',22,TEAL,'middle')
s.note('本文以 Paul 阱单离子的简谐久期运动为主；图与曲线均为原创教学示意。')
s.save('01-overview.svg')

# 2. Lorentzian-Gaussian convolution, fixed population (no probe backaction).
s=SVG('荧光谱线：热速度分布会把共振线摊宽','固定初始温度、弱束缚近似；369.5 nm、名义质量 171 u、Lorentzian FWHM = 20 MHz。')
a=Axis(s,100,158,620,325,(-50,50),(0,1.05),'普通频率失谐 Δν / MHz','相同探测效率下的相对散射率',[-50,-25,0,25,50],[0,.25,.5,.75,1])
freq=grid(-50,50,501)
z=grid(-8,8,801);dz=z[1]-z[0]
for T,color in [(0,INK),(.0005,BLUE),(.05,RED)]:
    sig=sqrt(K*T/M)/LAM/1e6
    yy=[1/(1+(f/10)**2) for f in freq] if T==0 else [trap([exp(-v*v/2)/sqrt(2*pi)/(1+((f-sig*v)/10)**2) for v in z],dz) for f in freq]
    a.plot(freq,yy,color)
for y,txt,color in [(196,'0 K：仅基线线宽',INK),(242,'0.5 mK：展宽很小',BLUE),(288,'50 mK：明显变宽',RED)]:
    s.line(760,y-6,800,y-6,color,3);s.text(812,y,txt,17,color)
s.text(758,359,'低温时，温度效应容易被',17)
s.text(758,389,'自然线宽、功率展宽与',17)
s.text(758,419,'激光频噪掩盖。',17)
s.note('这是静态两能级卷积模型；真实扫描还需处理冷却/加热、多能级和微运动。零温曲线不表示可实现的冷却状态。')
s.save('02-doppler.svg')

# 3. Imaging convolution in object coordinates.
s=SVG('空间成像：相机光斑 = 离子位置分布 ∗ 点扩散函数','名义 171 u；轴向频率 100 kHz；T = 0.5 mK；所有宽度均为物方 RMS。')
sx=sqrt(K*.0005/(M*(2*pi*1e5)**2))*1e6
psf=.45;obs=sqrt(sx*sx+psf*psf)
a=Axis(s,100,158,620,325,(-2.5,2.5),(0,1.8),'物方位置 x / μm','面积归一化的一维分布 / μm⁻¹',[-2,-1,0,1,2],[0,.5,1,1.5])
xx=grid(-2.5,2.5)
for sigma,color in [(sx,BLUE),(psf,GOLD),(obs,TEAL)]:
    a.plot(xx,[exp(-x*x/(2*sigma*sigma))/(sqrt(2*pi)*sigma) for x in xx],color)
for y,txt,col in [(194,f'离子 σx = {sx:.3f} μm',BLUE),(238,'PSF σ = 0.450 μm',GOLD),(282,f'观测 σ = {obs:.3f} μm',TEAL)]:
    s.line(756,y-6,795,y-6,col,3);s.text(807,y,txt,17,col)
s.text(758,354,'方差相减：',21,weight='bold');s.text(758,391,'σx² = σobs² − σPSF²',20)
s.text(758,437,'先统一物方 / 像方单位。',17)
s.note('星号表示卷积。PSF、漂移、像素采样与微运动都会影响实测宽度；高斯曲线为理想化示例。')
s.save('03-imaging.svg')

# 4. Deliberately qualitative recooling timeline, no invented temperature fit.
s=SVG('Doppler 再冷却：让荧光恢复过程报告加热后的能量','每次重新制备相同初态；重复“冷却 → 暗等待 → 短时间分箱探测”，累积光子计数。')
for x,w,txt,col in [(100,220,'预冷却',TEAL),(320,280,'暗等待 τ',GRAY),(600,400,'开冷却光并计数',BLUE)]:
    s.rect(x,120,w,62,col,col,3);s.text(x+w/2,158,txt,21,'#ffffff','middle')
s.text(100,218,'冷却光：开',17,TEAL);s.text(365,218,'冷却光：关',17,GRAY);s.text(650,218,'冷却光：开',17,BLUE)
a=Axis(s,100,284,620,205,(0,5),(0,1.08),'再冷却后的时间（任意单位）','荧光计数率（任意单位）',[0,1,2,3,4,5],[0,.5,1])
tt=grid(0,5)
for delay,col in [(.8,TEAL),(2.5,RED)]:
    yy=[.15+.85/(1+exp(-(t-delay)*4)) for t in tt]
    a.plot(tt,yy,col)
s.text(760,328,'较小初始能量：恢复较早',17,TEAL)
s.text(760,371,'较大初始能量：恢复较晚',17,RED)
s.text(760,426,'拟合对象：完整冷却动力学',17,weight='bold')
s.note('下方曲线仅用 sigmoid 示意典型恢复趋势，不是 Doppler 冷却方程的解，不能用于温度标定。')
s.save('04-recooling.svg')

# 5. Weak-probe first-order sideband strengths; carrier omitted.
s=SVG('边带不对称：没有声子，就不能再拿走一个声子','红边带消耗一个声子；蓝边带增加一个声子。下图仅画边带，载波省略。')
a=Axis(s,100,158,620,325,(-1.6,1.6),(0,1.05),'相对载波失谐 / 模频率 f','分别按各自蓝边带峰归一化',[-1,0,1],[0,.2,.4,.6,.8,1])
dd=grid(-1.6,1.6,801)
for nb,col in [(.2,BLUE),(5,RED)]:
    ratio=nb/(nb+1)
    yy=[ratio*exp(-.5*((d+1)/.09)**2)+exp(-.5*((d-1)/.09)**2) for d in dd]
    a.plot(dd,yy,col)
s.text(760,186,'n̄ = 0.2 → r = 1/6',19,BLUE)
s.text(760,228,'n̄ = 5 → r = 5/6',19,RED)
s.text(760,299,'−f：红边带',19,BLUE);s.text(760,339,'+f：蓝边带',19,TEAL)
s.text(760,403,'n̄ = r / (1 − r)',24,weight='bold')
s.note('弱探测、Lamb–Dicke 区且边带可分辨的理想谱；峰形为高斯示意。共同脉冲下的热态比值见正文推导。')
s.save('05-sidebands.svg')

# 6. Exact single-mode Laguerre matrix elements, ideal thermal mixture.
s=SVG('Rabi 振荡：不同声子数，对应不同的振荡频率','一阶蓝边带；η = 0.06；使用完整 Laguerre 矩阵元；无技术退相干。')
a=Axis(s,100,158,620,325,(0,5),(0,1.02),'脉冲时长 / n=0 蓝边带的 Rabi 周期','激发态概率 Pe',[0,1,2,3,4,5],[0,.25,.5,.75,1])
tt=grid(0,5,801);eta=.06
for nb,col in [(.1,BLUE),(2,TEAL),(10,RED)]:
    pp=[thermal(nb,n) for n in range(ncut(nb))]
    rates=[laguerre(n,1,eta*eta)/sqrt(n+1) for n in range(len(pp))]
    yy=[sum(p*sin(pi*t*r)**2 for p,r in zip(pp,rates)) for t in tt]
    a.plot(tt,yy,col,2.5)
for y,txt,col in [(191,'n̄ = 0.1',BLUE),(235,'n̄ = 2',TEAL),(279,'n̄ = 10',RED)]:
    s.line(760,y-6,800,y-6,col,3);s.text(815,y,txt,19,col)
s.text(760,354,'热布居的频率叠加',19,weight='bold');s.text(760,389,'产生塌缩与再相干。',18)
s.text(760,437,'包络衰减 ≠ 全部来自温度',17,RED)
s.note('曲线是纯粹的热态混合计算；实验拟合必须另外检查强度漂移、频率噪声和读出误差。')
s.save('06-rabi.svg')

# 7. Lambda system and explicitly phenomenological dark-resonance dip.
s=SVG('暗共振：用窄的双光子特征提高速度灵敏度','相干布居囚禁使荧光出现凹陷；两束光的波矢差决定双光子 Doppler 灵敏度。')
s.rect(48,126,410,385)
for x,y,txt in [(104,405,'|g₁〉'),(318,405,'|g₂〉'),(208,208,'|e〉')]:
    s.line(x,y,x+85,y,INK,3);s.text(x+42,y+30,txt,21,anchor='middle')
s.line(147,385,232,229,BLUE,3);s.line(348,385,270,229,TEAL,3)
s.text(111,289,'k₁',24,BLUE);s.text(339,289,'k₂',24,TEAL)
s.text(250,483,'δeff = δ − (k₁ − k₂)·v',20,anchor='middle')
a=Axis(s,550,185,470,275,(-4,4),(0,1.05),'双光子失谐 / 固有特征宽度','荧光（任意单位）',[-4,-2,0,2,4],[0,.5,1])
dd=grid(-4,4)
for sigma,col in [(0,BLUE),(1.1,RED)]:
    yy=[1-.85/(1+d*d) for d in dd] if sigma==0 else [1-.85*trap([exp(-v*v/2)/sqrt(2*pi)/(1+(d-sigma*v)**2) for v in z],dz) for d in dd]
    a.plot(dd,yy,col)
s.text(573,542,'蓝：无速度展宽　　红：加入高斯速度展宽',17)
s.note('三能级示意；凹陷采用现象学 Lorentzian 卷积，并非多能级 OBE 计算。不能凭本图为具体离子标定温度。')
s.save('07-dark-resonance.svg')

# 8. Kinetic energy is not always thermal energy.
s=SVG('微运动能量不等于热温度','Paul 阱中的 RF 驱动、相干久期振荡和随机热运动，需要分别建模。')
for i,title in enumerate(['随机热运动','相干久期运动','额外微运动']):
    x=40+354*i;s.rect(x,129,330,355);s.text(x+165,175,title,23,anchor='middle',weight='bold')
tt=grid(0,1,401)
for i,(cycles,col,caption) in enumerate([(3,BLUE,'热态：用分布描述'),(3,TEAL,'相位关联：需测位移 / 相干态'),(22,RED,'锁定 RF 相位：优先补偿')]):
    x=66+354*i
    # First panel: a few deterministic trajectories of random-phase/amplitude ensemble.
    if i==0:
        for amp,phase in [(.3,0),(.7,1),(.9,2.3),(.5,4.2)]:
            s.path([(x+t*270,307+68*amp*sin(2*pi*cycles*t+phase)) for t in tt],col,1.5)
    else:
        s.path([(x+t*270,307+63*sin(2*pi*cycles*t)) for t in tt],col,2)
    s.text(x+135,414,caption,16,col,'middle')
    s.text(x+135,452,'时间 →',17,GRAY,'middle')
s.rect(40,510,1038,53,'#e6f1f1','#c5e2df')
s.text(559,544,'同样的均方速度，可能对应完全不同的量子态与驱动机制。',21,TEAL,'middle')
s.note('轨迹仅为概念示意，幅度与频率不对应具体装置；第一栏用不同幅度和相位表示重复制备的热系综。')
s.save('08-motion.svg')

# Independent numerical checks of the equations and examples in the article.
checks={}
checks['constants']={'mass_u_nominal':171,'mode_frequency_Hz':F,'wavelength_m':LAM}
checks['hf_over_k_uK']=H*F/K*1e6
checks['doppler_fwhm_at_0p5mK_MHz']=sqrt(8*log(2)*K*.0005/M)/LAM/1e6
checks['position_rms_100kHz_0p5mK_um']=sx
checks['imaging_observed_rms_um']=obs
checks['imaging_roundtrip_T_mK']=M*(2*pi*1e5)**2*((obs*1e-6)**2-(psf*1e-6)**2)/K*1e3
checks['x_zero_point_1MHz_nm']=sqrt(HB/(2*M*W))*1e9
checks['examples']=[]
ratio_errors=[]
normalization_errors=[]
finite_pulse_errors=[]
moment_errors=[]
for nb in [.02,.2,1,5,20]:
    pp=[thermal(nb,n) for n in range(ncut(nb))]
    normalization_errors.append(abs(sum(pp)-1))
    mean=sum(n*p for n,p in enumerate(pp))
    moment_errors.append(abs(mean-nb))
    # Weak-pulse sideband strengths independently summed over Fock populations.
    ar=sum(n*p for n,p in enumerate(pp));ab=sum((n+1)*p for n,p in enumerate(pp))
    ratio=ar/ab
    ratio_errors.append(abs(ratio-nb/(1+nb)))
    temp=H*F/(K*log1p(1/nb))
    back=1/(exp(H*F/(K*temp))-1)
    assert abs(back-nb)<1e-10
    # Finite-pulse identity for a thermal state with matched isolated-sideband kernels.
    for time in [.3,1.4,3.7]:
        b=sum(p*sin(.5*time*laguerre(n,1,.01)/sqrt(n+1))**2 for n,p in enumerate(pp))
        r=sum(pp[n+1]*sin(.5*time*laguerre(n,1,.01)/sqrt(n+1))**2 for n in range(len(pp)-1))
        finite_pulse_errors.append(abs(r/b-nb/(1+nb)))
    checks['examples'].append({'nbar':nb,'ratio':ratio,'temperature_uK':temp*1e6})
checks['thermal_probability_max_error']=max(normalization_errors)
checks['thermal_mean_max_error']=max(moment_errors)
checks['weak_sideband_ratio_max_error']=max(ratio_errors)
checks['matched_finite_pulse_ratio_max_error']=max(finite_pulse_errors)
# Verify the Gaussian convolution variance directly by integrating the observed PDF.
xx=grid(-12*obs,12*obs,20001);dx=xx[1]-xx[0]
pdf=[exp(-x*x/(2*obs*obs))/(sqrt(2*pi)*obs) for x in xx]
second=trap([x*x*p for x,p in zip(xx,pdf)],dx)
checks['gaussian_variance_relative_error']=abs(second-obs*obs)/(obs*obs)
r=.16
checks['r_0p16_nbar']=r/(1-r)
checks['r_0p16_temperature_uK']=H*F/(-K*log(r))*1e6
checks['r_0p16_delta_r_0p02_delta_nbar']=.02/(1-r)**2
checks['r_0p16_delta_r_0p02_delta_T_uK']=H*F/K/(r*log(r)**2)*.02*1e6
checks['heating_50quanta_per_s_classical_uK_per_s']=H*F/K*50*1e6
checks['nonthermal_example']={'p0':.99,'p100':.01,'mean_n':1.,'ground_population':.99,'thermal_ground_population_at_mean_1':.5}
checks['figure_count']=len(list(OUT.glob('*.svg')))
assert max(normalization_errors)<1e-11
assert max(moment_errors)<1e-9
assert max(ratio_errors)<1e-11
assert max(finite_pulse_errors)<1e-10
assert checks['gaussian_variance_relative_error']<1e-9
assert abs(checks['imaging_roundtrip_T_mK']-.5)<1e-12
assert checks['figure_count']==8
(HERE/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
print(f'Created eight SVG figures in {OUT}')
