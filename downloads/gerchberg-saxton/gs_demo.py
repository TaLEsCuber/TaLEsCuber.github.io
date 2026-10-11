"""Reproduce the GS tutorial figures and numerical checks.

Python 3.10+, NumPy, Matplotlib. Run: python gs_demo.py
Default output is ./gs-output; in this repository use --output source/img/gerchberg-saxton
and --report source/downloads/gerchberg-saxton/validation.json.
All fields use a unitary, centered DFT; the example is a simulation, not hardware data.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse


def ft(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm='ortho'))


def ift(v):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(v), norm='ortho'))


def project(z, amplitude):
    # angle(0)=0 is one allowed choice where phase is undefined.
    return amplitude * np.exp(1j * np.angle(z))


def make_problem(n=256):
    q = np.arange(n) - n // 2
    x, y = np.meshgrid(q, q)
    a = np.exp(-(x*x + y*y) / 65**2) * (x*x + y*y <= 100**2)
    a /= np.linalg.norm(a)
    target = sum(np.exp(-((x-cx)**2+(y-cy)**2)/(2*3.5**2))
                 for cy in (-28, 0, 28) for cx in (-28, 0, 28))
    target /= target.sum()
    return a, np.sqrt(target), x, y


def gs(a, b, steps=300, seed=7):
    rng = np.random.default_rng(seed)
    u = a * np.exp(1j*rng.uniform(-np.pi, np.pi, a.shape))
    errors, snapshots = [], {}
    for k in range(steps + 1):
        v = ft(u)
        errors.append(float(np.linalg.norm(np.abs(v)-b)/np.linalg.norm(b)))
        if k in (0, 1, 10, 50, steps):
            snapshots[k] = np.abs(v)**2
        if k < steps:
            u = project(ift(project(v, b)), a)
    return u, np.asarray(errors), snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('gs-output'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False, 'font.size':12, 'axes.spines.top':False,
        'axes.spines.right':False, 'figure.facecolor':'#f6f8fc',
        'axes.facecolor':'#f6f8fc', 'savefig.facecolor':'#f6f8fc'})
    blue, orange, teal = '#2463a6', '#d47a20', '#168c89'
    def save(fig, name):
        fig.savefig(args.output / name, dpi=170, bbox_inches='tight')
        plt.close(fig)
    def box(ax, xy, w, h, txt, color=blue, size=13):
        ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.025',
                     facecolor='white',edgecolor=color,linewidth=2))
        ax.text(xy[0]+w/2,xy[1]+h/2,txt,ha='center',va='center',fontsize=size,color=color)
    def arrow(ax, start, end, color=blue):
        ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':color,'lw':2})

    a,b,x,y = make_problem()
    u,err,snaps = gs(a,b)
    intensity = np.abs(ft(u))**2
    fig,axs = plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,arr,title,cmap in zip(axs,[a*a,np.ma.masked_where(a==0,np.angle(u)),b*b],
            ['输入面：已知光强 $a^2$','需要设计：输入相位 $\\phi$','目标面：指定光强 $b^2$'],
            ['magma','twilight','magma']):
        palette=plt.get_cmap(cmap).copy()
        palette.set_bad('#dfe5ed')
        im=ax.imshow(arr,cmap=palette,origin='lower');ax.set_title(title);ax.set_axis_off()
        if cmap=='twilight':
            im.set_clim(-np.pi,np.pi)
            fig.colorbar(im,ax=ax,shrink=.65,ticks=[-np.pi,0,np.pi]).ax.set_yticklabels(['−π','0','π'])
    fig.suptitle('GS：在两个平面之间，寻找能够协调振幅约束的相位',fontsize=17)
    save(fig,'01-two-planes.png')

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ax=axs[0]; ax.add_patch(Circle((0,0),1,fill=False,color=teal,lw=2))
    theta=.68; z=1.6*np.exp(1j*theta); p=np.exp(1j*theta)
    ax.axhline(0,color='#aab4c5',lw=1);ax.axvline(0,color='#aab4c5',lw=1)
    arrow(ax,(0,0),(z.real,z.imag));ax.plot([p.real,z.real],[p.imag,z.imag],color=orange,lw=4)
    ax.scatter([z.real,p.real],[z.imag,p.imag],color=[blue,teal],s=70,zorder=5)
    ax.text(z.real+.02,z.imag+.12,'当前复数 $z=re^{i\\theta}$',fontsize=12)
    ax.text(p.real-.95,p.imag+.2,'投影 $be^{i\\theta}$',color=teal)
    ax.text(-1.05,-1.32,'保留角度 θ，只把半径 r 改成 b',color=blue)
    ax.set(xlim=(-1.5,2),ylim=(-1.6,1.7),xlabel='实部',ylabel='虚部',title='单个像素：最近点振幅投影')
    ax.set_aspect('equal')
    ax=axs[1];ax.set_axis_off()
    box(ax,(.05,.65),.88,.21,'固定振幅集合是圆周，不是圆盘',teal,15)
    ax.text(.08,.49,'两个允许点的平均值可能落在圆内，\n因此不再满足固定振幅约束。',va='center',fontsize=14)
    ax.text(.08,.23,'约束非凸 → 可能停滞、依赖初值\n误差下降 ≠ 得到唯一真实相位',va='center',fontsize=15,color=orange)
    save(fig,'02-amplitude-projection.png')

    fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    box(ax,(.4,4.1),4,1.15,'① 输入面满足约束\n$u_k=a\\exp(i\\phi_k)$')
    box(ax,(7.5,4.1),4,1.15,'② 正向传播后，先计算误差\n$v_k=\\mathcal{F}u_k$')
    box(ax,(7.5,.7),4,1.15,'③ 目标面替换振幅\n$\\hat v_k=b\\exp(i\\arg v_k)$',teal)
    box(ax,(.4,.7),4,1.15,'④ 反向传播，再恢复输入振幅\n$u_{k+1}=a\\exp(i\\arg\\mathcal{F}^{-1}\\hat v_k)$',teal)
    arrow(ax,(4.6,4.7),(7.3,4.7));ax.text(5.95,5,'FFT',ha='center',color=blue)
    arrow(ax,(9.5,3.95),(9.5,2));ax.text(9.8,3,'保留相位',color=teal)
    arrow(ax,(7.3,1.3),(4.6,1.3));ax.text(5.95,.65,'IFFT',ha='center',color=blue)
    arrow(ax,(2.4,2),(2.4,3.95));ax.text(.65,3,'继续迭代',color=teal)
    ax.text(6,3,'误差必须在步骤②测量\n步骤③的振幅是人为强制正确的',ha='center',va='center',color=orange,fontsize=14)
    ax.set_title('一次 GS 迭代：传播 → 振幅投影 → 逆传播 → 振幅投影',fontsize=17)
    save(fig,'03-iteration-loop.png')

    fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    arrays=[b*b]+[snaps[k] for k in (0,1,10,50,300)]
    titles=['目标强度']+[f'{k} 次更新后' for k in (0,1,10,50,300)]
    for ax,arr,title in zip(axs.flat,arrays,titles):
        im=ax.imshow(arr,cmap='magma',origin='lower',vmin=0,vmax=(b*b).max(),extent=(-128,127,-128,127))
        ax.set(xlim=(-60,60),ylim=(-60,60),title=title,xlabel='傅里叶面像素',ylabel='傅里叶面像素')
    fig.colorbar(im,ax=list(axs.flat),shrink=.7,extend='max',label='强度（总功率归一为 1，所有子图同一色标）')
    fig.suptitle('九光斑全息图的演化｜256 × 256 网格，随机种子 7',fontsize=17)
    save(fig,'04-reconstruction.png')

    histories={7:err}; seed_results={7:float(err[-1])}
    for seed in (1,11,23):
        _,e,_=gs(a,b,seed=seed);histories[seed]=e;seed_results[seed]=float(e[-1])
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for seed,e in histories.items():
        axs[0].semilogy(np.arange(len(e)),e,label=f'seed = {seed}')
    axs[0].set(xlabel='完成的 GS 更新次数',ylabel='相对振幅误差 $E_A$',title='先快速下降，再逐渐进入平台')
    axs[0].grid(alpha=.2);axs[0].legend()
    axs[1].plot(x[128],(b*b)[128],label='目标强度',color=blue,lw=2)
    axs[1].plot(x[128],intensity[128],label='实际重建（seed 7）',color=orange,lw=1.4)
    axs[1].set(xlim=(-60,60),xlabel='傅里叶面横向像素',ylabel='归一化强度',title='中心横截面：残差并未消失')
    axs[1].legend();axs[1].grid(alpha=.2)
    save(fig,'05-error-and-profile.png')

    fig,ax=plt.subplots(figsize=(12,5.6));ax.set(xlim=(0,12),ylim=(0,5.5));ax.axis('off')
    ax.plot([.4,11.7],[2.7,2.7],ls='--',color='#aab4c5')
    ax.plot([2,2],[1.3,4.1],color=blue,lw=7)
    ax.add_patch(Ellipse((6,2.7),.35,3.2,facecolor='#d6eff5',edgecolor=teal,lw=2))
    ax.plot([10,10],[1.1,4.3],color=orange,lw=4)
    for yy in (1.7,2.2,3.2,3.7):
        ax.plot([2,6,10],[yy,yy,2.7],color=teal,alpha=.4)
    ax.text(2,4.65,'输入面 / SLM\n固定振幅，加载相位',ha='center',color=blue,fontsize=14)
    ax.text(6,4.65,'薄透镜',ha='center',color=teal,fontsize=14)
    ax.text(10,4.65,'后焦面 / 相机\n测量重建强度',ha='center',color=orange,fontsize=14)
    for xx1,xx2 in ((2,6),(6,10)):
        ax.annotate('',(xx1,.9),(xx2,.9),arrowprops={'arrowstyle':'<->','color':blue})
        ax.text((xx1+xx2)/2,.5,'f',ha='center',fontsize=15)
    ax.text(6,0,'理想前焦面—后焦面傅里叶变换；光线仅示意，不代表九光斑逐点传播',ha='center',fontsize=12)
    ax.set_title('从数值相位到实验：坐标缩放与器件响应同样重要',fontsize=17)
    save(fig,'06-optical-system.png')

    rng=np.random.default_rng(2026)
    checks={}
    for n in (7,8,31,32):
        z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        checks[f'fft_roundtrip_{n}']=float(np.linalg.norm(ift(ft(z))-z)/np.linalg.norm(z))
        checks[f'parseval_{n}']=float(abs(np.linalg.norm(ft(z))**2/np.linalg.norm(z)**2-1))
    n=7; z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    d=np.exp(-2j*np.pi*np.outer(np.arange(n),np.arange(n))/n)/np.sqrt(n)
    direct=np.fft.fftshift(d@np.fft.ifftshift(z)@d.T)
    checks['direct_dft_relative_error']=float(np.linalg.norm(direct-ft(z))/np.linalg.norm(direct))
    checks['source_amplitude_max_error']=float(np.max(np.abs(np.abs(u)-a)))
    checks['power_relative_error']=float(abs(intensity.sum()/(a*a).sum()-1))
    checks['max_error_increase_all_seeds']=float(max(np.max(np.diff(e)) for e in histories.values()))
    checks['global_phase_intensity_max_error']=float(np.max(np.abs(np.abs(ft(u*np.exp(.71j)))**2-intensity)))
    # Verify the projection-distance identity independently on arbitrary complex arrays.
    z=rng.normal(size=(19,19))+1j*rng.normal(size=(19,19));amp=rng.random((19,19))
    checks['projection_distance_identity_error']=float(abs(np.linalg.norm(z-project(z,amp))**2-np.linalg.norm(np.abs(z)-amp)**2))
    checks['zero_projection_finite']=bool(np.isfinite(project(np.zeros((3,3)),np.ones((3,3)))).all())
    roi=sum(((x-cx)**2+(y-cy)**2<=8**2).astype(int) for cy in (-28,0,28) for cx in (-28,0,28))>0
    powers=[float(intensity[(x-cx)**2+(y-cy)**2<=8**2].sum()) for cy in (-28,0,28) for cx in (-28,0,28)]
    results={'model':'Unitary centered DFT, ideal scalar monochromatic phase-only hologram',
        'n':256,'steps':300,'seed':7,'input_amplitude_waist_pixels':65,'aperture_radius_pixels':100,
        'target_intensity_sigma_pixels':3.5,'target_centers_pixels':[-28,0,28],
        'amplitude_errors':{str(k):float(err[k]) for k in (0,1,10,50,300)},
        'final_relative_intensity_error':float(np.linalg.norm(intensity-b*b)/np.linalg.norm(b*b)),
        'roi_radius_pixels':8,'roi_efficiency':float(intensity[roi].sum()/intensity.sum()),
        'target_power_fraction_in_roi':float((b*b)[roi].sum()),
        'spot_integrated_power_cv':float(np.std(powers)/np.mean(powers)),
        'seed_final_amplitude_errors':seed_results,'checks':checks}
    assert max(v for k,v in checks.items() if k.startswith(('fft_roundtrip','parseval','direct_dft'))) < 1e-12
    assert checks['max_error_increase_all_seeds'] < 1e-12
    assert checks['source_amplitude_max_error'] < 1e-14
    assert checks['power_relative_error'] < 1e-12
    assert checks['global_phase_intensity_max_error'] < 1e-14
    assert checks['projection_distance_identity_error'] < 1e-10
    assert checks['zero_projection_finite']
    report=args.report or args.output/'validation.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(report.parent/'gs-result.npz',input_amplitude=a,target_amplitude=b,
                        phase=np.angle(u),reconstructed_intensity=intensity,error=err)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
