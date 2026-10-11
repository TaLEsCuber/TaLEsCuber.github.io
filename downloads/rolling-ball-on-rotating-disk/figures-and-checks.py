"""Original figures and independent ODE checks for the rotating-disk article.

Requirements: Python 3, numpy, scipy, matplotlib. Run from any directory:
    python figures-and-checks.py
In the blog checkout, figures go to source/img/rolling-ball-on-rotating-disk.
For a standalone download, they go to ./rolling-ball-figures beside this script.
All curves are ideal-model calculations, not experimental measurements.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent
OUT = SOURCE / 'img/rolling-ball-on-rotating-disk' if SOURCE.name == 'source' else HERE / 'rolling-ball-figures'
OUT.mkdir(parents=True, exist_ok=True)
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei'] if f in available), 'DejaVu Sans')
plt.rcParams.update({'font.family': font, 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#23354d', 'text.color': '#23354d',
                     'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#ffffff',
                     'savefig.facecolor': '#f7f9fc', 'mathtext.fontset': 'dejavusans'})
BLUE, ORANGE, GREEN, GREY = '#1874ad', '#df6b35', '#158575', '#8493a3'
J = np.array([[0., -1.], [1., 0.]])

def arrow(ax, start, end, color=BLUE, label=None, textpos=None, **kw):
    kw.setdefault('zorder', 4)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
                                linewidth=2, color=color, **kw))
    if label:
        p = textpos if textpos is not None else (np.asarray(start)+np.asarray(end))/2
        ax.text(*p, label, color=color, fontsize=12)

def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)

def orbit(t, r0, v0, omega):
    r0, v0 = np.asarray(r0), np.asarray(v0)
    th = omega*np.asarray(t)
    v = np.cos(th)[:, None]*v0 + np.sin(th)[:, None]*(J@v0)
    r = r0 + np.sin(th)[:, None]*v0/omega + (1-np.cos(th))[:, None]*(J@v0)/omega
    return r, v

def disk_axes(ax, R=.15):
    ax.add_patch(Circle((0,0), R, color='#edf2f7', ec=GREY, lw=1.5))
    ax.axhline(0, color='#d6dfe7', lw=.7); ax.axvline(0, color='#d6dfe7', lw=.7)
    ax.plot(0,0,'+',color='#23354d',ms=9)
    ax.set_aspect('equal'); ax.set_xlim(-R*1.17,R*1.17); ax.set_ylim(-R*1.17,R*1.17)
    ax.set_xlabel('x / m'); ax.set_ylabel('y / m')

# 1. Geometry and actual forces. Top view is an off-centre orbit.
fig, axs = plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
ax=axs[0]; ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.3,1.4); ax.set_ylim(-1.25,1.4)
ax.add_patch(Circle((0,0),1.05,fc='#eaf0f7',ec=GREY,lw=2))
c=np.array([.22,.15]); rho=.52; th=np.linspace(0,2*np.pi,400)
ax.plot(c[0]+rho*np.cos(th),c[1]+rho*np.sin(th),'--',color=BLUE,lw=2)
p=c+rho*np.array([np.cos(.55),np.sin(.55)])
ax.plot(0,0,'k+'); ax.text(-.18,-.13,'O：盘心'); ax.plot(*c,'o',color=GREEN,ms=5); ax.text(c[0]-.15,c[1]-.18,'C：轨道圆心',fontsize=10)
ax.add_patch(Circle(p,.075,fc=ORANGE,ec='white',zorder=5))
arrow(ax,p,p+.46*np.array([-np.sin(.55),np.cos(.55)]),BLUE,'速度 v',(.58,.91))
arrow(ax,p,p+.34*(c-p)/rho,ORANGE,'静摩擦 f',(.52,.14))
arrow(ax,(0,0),p,GREY,'r',(.39,.27))
arrow(ax,(-.77,.48),(-.92,.15),GREEN,connectionstyle='arc3,rad=.25')
ax.text(-1.02,.65,'Ω > 0\n逆时针',color=GREEN,fontsize=11)
ax.set_title('俯视：摩擦指向轨道圆心 C',fontweight='bold',pad=15)
ax=axs[1]; ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.4,1.5); ax.set_ylim(-.55,1.8)
ax.plot([-1.3,1.3],[0,0],color=GREY,lw=5)
ax.add_patch(Circle((0,.58),.58,fc='#e4edf7',ec=BLUE,lw=2)); ax.plot(0,.58,'o',color=BLUE)
arrow(ax,(0,.58),(0,1.58),GREEN,'N',(.06,1.3)); arrow(ax,(0,.58),(0,-.39),ORANGE,'mg',(.08,-.36))
arrow(ax,(0,0),(-.94,0),ORANGE,'f',(-.85,.14))
arrow(ax,(.18,.58),(.18,.03),GREY,'a',(.25,.28))
ax.text(.17,.69,'球心'); ax.text(.15,-.23,'接触点 P'); ax.text(-1.27,-.48,'水平：m dv/dt = f     竖直：N = mg',fontsize=11)
ax.set_title('侧视：同一摩擦力同时产生力矩',fontweight='bold',pad=15)
fig.suptitle('01  小球的平动与转动必须一起计算',fontsize=19,fontweight='bold')
save(fig,'01-model.png')

# 2. Exact contact velocity triangle in the common planar coordinates.
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
ax=axs[0];ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-.2,3.8);ax.set_ylim(-.6,2.5)
arrow(ax,(0,0),(1.25,1.55),BLUE,'球心速度 v',(.02,1.25))
arrow(ax,(1.25,1.55),(3.1,.65),ORANGE,'自转贡献 a k × ω', (1.78,1.47))
arrow(ax,(0,0),(3.1,.65),GREEN,'盘面速度 u', (1.25,-.25))
ax.text(.1,2.1,'接触点速度相加，而不是令球心追上盘面',fontsize=12)
ax.set_title('一般情况：v + a k × ω = u',fontweight='bold')
ax=axs[1];ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.5,1.6);ax.set_ylim(-.5,2.2)
ax.plot([-1.3,1.3],[0,0],color=GREY,lw=4);ax.add_patch(Circle((0,.62),.62,ec=BLUE,fc='#e4edf7',lw=2))
ax.plot(0,.62,'o',color=BLUE);ax.text(.15,.72,'v = 0')
arrow(ax,(-.95,-.22),(.95,-.22),GREEN,'运动的盘面 u',(-.75,-.49))
arrow(ax,(0,0),(.93,0),ORANGE)
arrow(ax,(-.48,.99),(-.57,.55),ORANGE,connectionstyle='arc3,rad=.22')
ax.text(-1.35,1.66,'只要预先给球合适的自转，\n球心就能固定在实验室的某一点。',fontsize=12)
ax.set_title('特殊情况：原地滚动',fontweight='bold')
fig.suptitle('02  无滑动 ≠ 接触点在实验室中静止',fontsize=19,fontweight='bold')
save(fig,'02-contact-velocity.png')

Omega=2*np.pi; kappa=.4; omega=kappa/(1+kappa)*Omega; T=2*np.pi/omega
t=np.linspace(0,T,800)
fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
cases=[([0,0],[.10,0],'从盘心向右出发'),([.06,0],[0,omega*.06],'特意选成绕盘心的圆'),([.04,-.01],[.07,.025],'一般初始条件：偏心圆')]
for ax,(r0,v0,title) in zip(axs,cases):
    disk_axes(ax);r,v=orbit(t,r0,v0,omega);c=np.asarray(r0)+J@v0/omega
    ax.plot(r[:,0],r[:,1],color=BLUE,lw=2.5); ax.plot(*r0,'o',color=ORANGE,label='起点')
    ax.plot(*c,'x',color=GREEN,ms=9,label='轨道圆心 C')
    arrow(ax,r[120],r[140],BLUE);ax.set_title(title,fontsize=13,fontweight='bold')
    ax.legend(loc='lower left',fontsize=9)
fig.suptitle('03  三条不同圆轨道，共同周期 T = 3.5 s（转盘 60 rpm）',fontsize=18,fontweight='bold')
save(fig,'03-orbits.png')

tt=np.linspace(0,7,1800); r0=[.04,-.01];v0=[.07,.025];r,v=orbit(tt,r0,v0,omega)
angles=Omega*tt
q=np.column_stack((np.cos(angles)*r[:,0]+np.sin(angles)*r[:,1],-np.sin(angles)*r[:,0]+np.cos(angles)*r[:,1]))
fig,axs=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
for ax,data,title in zip(axs,[r,q],['实验室：绕固定 C 转两圈','随盘旋转：两种旋转叠加成花形']):
    disk_axes(ax);ax.plot(data[:,0],data[:,1],color=BLUE,lw=1.8)
    ax.plot(*data[0],'o',color=ORANGE);ax.set_title(title,fontweight='bold')
axs[0].plot(*(np.asarray(r0)+J@v0/omega),'x',color=GREEN,ms=10)
axs[1].set_xlabel('X / m');axs[1].set_ylabel('Y / m')
fig.suptitle('04  同一运动，两个参考系：展示 7 个转盘周期',fontsize=18,fontweight='bold')
save(fig,'04-reference-frames.png')

fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
kk=np.linspace(0,1,400);ax.plot(kk,kk/(1+kk),color=BLUE,lw=3)
for k,y,lab,xy in [(.4,2/7,'均匀实心球：2/7',(.48,.22)),(2/3,2/5,'均匀薄球壳：2/5',(.69,.35))]:
    ax.plot(k,y,'o',color=ORANGE,ms=8);ax.vlines(k,0,y,colors=GREY,linestyles='--');ax.hlines(y,0,k,colors=GREY,linestyles='--')
    ax.annotate(lab,(k,y),xytext=xy,arrowprops={'arrowstyle':'->','color':ORANGE},color=ORANGE)
ax.set_xlim(0,1);ax.set_ylim(0,.54);ax.set_xlabel(r'无量纲转动惯量 $\kappa=I/(ma^2)$');ax.set_ylabel(r'角速度比 $\omega_c/\Omega$');ax.grid(alpha=.18)
ax.set_title('05  决定频率比的是质量分布：ωc / Ω = κ / (1 + κ)',fontsize=17,fontweight='bold')
save(fig,'05-inertia.png')

m=.02;a=.01;r,v=orbit(t,[0,0],[.1,0],omega)
wh=(Omega*r+v@J.T)/a
Etr=.5*m*np.sum(v*v,axis=1); Erot=.5*kappa*m*a*a*np.sum(wh*wh,axis=1)
force=m*omega*(v@J.T);power=np.sum(force*(Omega*(r@J.T)),axis=1)
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
axs[0].plot(t,Etr*1000,label='平动动能',color=BLUE,lw=2.5);axs[0].plot(t,Erot*1000,label='转动动能',color=ORANGE,lw=2.5);axs[0].plot(t,(Etr+Erot)*1000,label='总动能',color=GREEN,ls='--',lw=2)
axs[0].set_ylabel('能量 / mJ');axs[0].legend(fontsize=10);axs[0].set_title('速率不变，总动能仍可变化')
axs[1].plot(t,power*1000,color=ORANGE,lw=2.5);axs[1].axhline(0,color=GREY,lw=1);axs[1].fill_between(t,0,power*1000,where=power>0,color=ORANGE,alpha=.14);axs[1].fill_between(t,0,power*1000,where=power<0,color=BLUE,alpha=.14)
axs[1].set_ylabel('盘面对小球的功率 / mW');axs[1].set_title('前半周期输入能量，后半周期收回')
for ax in axs:ax.set_xlabel('t / s');ax.grid(alpha=.18)
fig.suptitle('06  静摩擦可以传能：P = f · u（示例取竖直自转为零）',fontsize=17,fontweight='bold')
save(fig,'06-energy.png')

tilt=np.deg2rad(.05);gp=np.array([9.81*np.sin(tilt),0.]);vd=J@gp/(kappa*Omega)
td=np.linspace(0,10.5,1800);rr,vv=orbit(td,[0,0],np.array([.055,0])-vd,omega);rr+=td[:,None]*vd
fig,axs=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
ax=axs[0];ax.plot(rr[:,0]*100,rr[:,1]*100,color=BLUE,lw=2);c=J@(np.array([.055,0])-vd)/omega
ax.plot((c[0]+td*vd[0])*100,(c[1]+td*vd[1])*100,'--',color=GREEN,label='移动的轨道圆心')
ax.set_aspect('equal');ax.set_xlabel('x / cm');ax.set_ylabel('y / cm');ax.set_title('倾斜 0.05°：圆周运动叠加缓慢漂移');ax.legend(fontsize=10);ax.grid(alpha=.18)
ax=axs[1];ax.axis('off');ax.set_xlim(-.2,2);ax.set_ylim(-.4,1.7)
arrow(ax,(.2,.1),(1.55,.1),ORANGE,'重力沿盘面分量 g∥',(.35,-.14))
arrow(ax,(.2,.1),(.2,1.4),GREEN,'平均漂移 vd',(.35,1.16))
ax.text(.75,.62,'Ω 沿盘面法线向外\n平均漂移垂直于下坡方向\n\n本例 vd ≈ 3.41 mm/s',fontsize=12)
fig.suptitle('07  盘面不水平时，轨道圆心会漂移',fontsize=18,fontweight='bold')
save(fig,'07-tilt-drift.png')

# Numerical verification independent of the closed-form position functions.
# Integrate translation and rotation separately under the Newton-Euler force.
rng=np.random.default_rng(20261005)
errors={'position_m':0.,'rolling_constraint_m_per_s':0.,'speed_m_per_s':0.,'orbit_radius_m':0.,'energy_power_W':0.,'rotating_frame_m':0.,'tilt_m':0.}
cases_checked=0
for kap in [.08,.4,2/3,.93]:
    for Om in [-8.,.15,2*np.pi]:
        for _ in range(3):
            mass=float(rng.uniform(.005,.2));rad=float(rng.uniform(.005,.04)); I=kap*mass*rad**2
            om=kap/(1+kap)*Om; per=2*np.pi/abs(om)
            r0=rng.uniform(-.07,.07,2);v0=rng.uniform(-.12,.12,2); w0=(Om*r0+J@v0)/rad
            def rhs(t,y):
                f=mass*om*(J@y[2:4])
                return np.r_[y[2:4],f/mass,-rad/I*(J@f)]
            times=np.linspace(0,2*per,241)
            sol=solve_ivp(rhs,[0,2*per],np.r_[r0,v0,w0],t_eval=times,rtol=2e-11,atol=2e-12)
            assert sol.success
            exact,vel=orbit(times,r0,v0,om); rr=sol.y[:2].T;vv=sol.y[2:4].T;ww=sol.y[4:6].T
            errors['position_m']=max(errors['position_m'],float(np.max(np.linalg.norm(rr-exact,axis=1))))
            errors['rolling_constraint_m_per_s']=max(errors['rolling_constraint_m_per_s'],float(np.max(np.linalg.norm(vv+rad*(ww@J.T)-Om*(rr@J.T),axis=1))))
            errors['speed_m_per_s']=max(errors['speed_m_per_s'],float(np.max(abs(np.linalg.norm(vv,axis=1)-np.linalg.norm(v0)))))
            c=r0+J@v0/om
            errors['orbit_radius_m']=max(errors['orbit_radius_m'],float(np.max(abs(np.linalg.norm(rr-c,axis=1)-np.linalg.norm(v0)/abs(om)))))
            f=mass*om*(vv@J.T);dw=-rad/I*(f@J.T)
            dE=np.sum(f*vv,axis=1)+I*np.sum(ww*dw,axis=1)
            P=np.sum(f*(Om*(rr@J.T)),axis=1)
            errors['energy_power_W']=max(errors['energy_power_W'],float(np.max(abs(dE-P))))
            cases_checked+=1

# Rotating-frame equation and tilt forcing checked with separately integrated ODEs.
for kap in [.4,2/3]:
    om=kap/(1+kap)*Omega;alpha=kap/(1+kap);r0=np.array([.04,-.01]);v0=np.array([.07,.025]);times=np.linspace(0,7,801)
    def rotating_rhs(t,y):
        return np.r_[y[2:],-(2-alpha)*Omega*(J@y[2:])+(1-alpha)*Omega**2*y[:2]]
    sol=solve_ivp(rotating_rhs,[0,7],np.r_[r0,v0-Omega*J@r0],t_eval=times,rtol=1e-11,atol=1e-12)
    r,_=orbit(times,r0,v0,om);ct=np.cos(Omega*times);st=np.sin(Omega*times)
    q=np.c_[ct*r[:,0]+st*r[:,1],-st*r[:,0]+ct*r[:,1]]
    errors['rotating_frame_m']=max(errors['rotating_frame_m'],float(np.max(abs(sol.y[:2].T-q))))
    b=gp/(1+kap);vd=J@b/om
    sol=solve_ivp(lambda t,y:np.r_[y[2:],om*(J@y[2:])+b],[0,7],np.r_[r0,v0],t_eval=times,rtol=1e-11,atol=1e-12)
    r,_=orbit(times,r0,v0-vd,om);r+=times[:,None]*vd
    errors['tilt_m']=max(errors['tilt_m'],float(np.max(abs(sol.y[:2].T-r))))

# Moment of inertia from an independent cylindrical-slice integral, m=a=1.
density=3/(4*np.pi)
I_integral=quad(lambda z: .5*np.pi*density*(1-z*z)**2,-1,1,epsabs=1e-13)[0]
assert abs(I_integral-.4)<1e-13
assert all(e<3e-8 for e in errors.values()),errors
report={'model':'ideal point-contact, isotropic sphere, constant driven disk speed',
        'newton_euler_cases':cases_checked,'samples_per_case':241,'max_absolute_errors':errors,
        'solid_sphere_inertia_integral':I_integral,
        'example':{'disk_rpm':60,'Omega_rad_s':Omega,'omega_c_rad_s':omega,'T_orbit_s':T,
                   'v0_m_s':.1,'rho_m':.1/omega,'max_distance_from_disk_center_m':.2/omega,
                   'static_mu_min':omega*.1/9.81,'tilt_drift_mm_s':float(np.linalg.norm(J@gp/(.4*Omega))*1000)},
        'figures':sorted(p.name for p in OUT.glob('*.png'))}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
