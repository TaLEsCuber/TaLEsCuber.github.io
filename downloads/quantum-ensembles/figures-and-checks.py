"""Reproduce the quantum-ensemble article's figures and numerical checks.

Run: python source/downloads/quantum-ensembles/figures-and-checks.py
Dependencies: numpy, scipy, matplotlib. No experimental data are used.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.special import expit, softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'source/img/quantum-ensembles'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#f8fafc', 'axes.facecolor': '#f8fafc', 'text.color': '#19334a', 'axes.labelcolor': '#19334a', 'svg.fonttype': 'path'})
BLUE, TEAL, ORANGE, INK = '#377bb5', '#098779', '#dc8535', '#19334a'

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)

def box(ax, xy, w, h, label, color=BLUE):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.025', facecolor='white', edgecolor=color, linewidth=2))
    ax.text(xy[0]+w/2, xy[1]+h/2, label, ha='center', va='center', fontsize=14, linespacing=1.7)

fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
for ax, title, sub, desc in zip(axes, ['微正则系综', '正则系综', '巨正则系综'], ['固定 E、V、N', '固定 T、V、N', '固定 T、V、μ'], ['隔离边界\n无能量、粒子交换', '热库\n可以交换能量', '热库 + 粒子库\n交换能量与粒子']):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.set_title(title+'\n'+sub, fontsize=18, weight='bold', pad=16)
    box(ax, (.16, .42), .68, .27, '研究系统\n'+('能壳内等权' if ax is axes[0] else '能量会涨落' if ax is axes[1] else '能量、粒子数会涨落'))
    ax.text(.5, .15, desc, ha='center', va='center', fontsize=13, linespacing=1.6)
axes[0].plot([.1, .9], [.35, .35], color=INK, lw=4)
for ax in axes[1:]:
    ax.annotate('', (.37, .42), (.37, .27), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2.5))
    ax.text(.39, .32, '能量', fontsize=11, color=ORANGE)
axes[2].annotate('', (.73, .42), (.73, .27), arrowprops=dict(arrowstyle='<->', color=TEAL, lw=2.5))
axes[2].text(.75, .32, '粒子', fontsize=11, color=TEAL)
fig.suptitle('同一种量子语言，不同的宏观约束', fontsize=22, weight='bold')
save(fig, '01-ensemble-map')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
states = [np.array([[.5,.5],[.5,.5]]), np.eye(2)/2, np.diag([.206947,.793053])]
titles = ['相干叠加态 |+x〉', '等概率混合态', 'z 方向磁场中的热态']
notes = ['纯度 = 1；测量 σx 必得 +1', '纯度 = 1/2；测量 σx 得 ±1 各半', '纯度 ≈ 0.672；低能态占据更多']
for ax, mat, title, note in zip(axes, states, titles, notes):
    ax.imshow(mat, vmin=0, vmax=1, cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=20, color='white' if mat[i,j]>.65 else INK)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['↑','↓'], yticklabels=['↑','↓'])
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel(note, fontsize=11, labelpad=14)
fig.suptitle('同在 Sz 基底中：相同的对角元，不代表相同的量子态', fontsize=19, weight='bold')
save(fig, '02-density-matrices')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
ax=axes[0]
levels=[(-2,1),(0,2),(2,1)]
for en,g in levels:
    for j in range(g): ax.plot([j*.6-.35,j*.6+.1],[en,en],color=TEAL if en==0 else BLUE,lw=4)
ax.axhspan(-.4,.4,color=TEAL,alpha=.12)
ax.text(.9,2,'|↑↑〉：E = +2a',va='center'); ax.text(.9,0,'|↑↓〉、|↓↑〉：E = 0',va='center'); ax.text(.9,-2,'|↓↓〉：E = −2a',va='center')
ax.set(xlim=(-.6,3),ylim=(-2.8,2.8),xticks=[],yticks=[-2,0,2],ylabel='总能量 / a')
ax.set_title('两个无相互作用自旋：选择 E = 0 能壳',loc='left',weight='bold')
ax=axes[1]
mat=np.diag([0,.5,.5,0]); ax.imshow(mat,cmap='Greens',vmin=0,vmax=.6)
for i in range(4):
    for j in range(4): ax.text(j,i,f'{mat[i,j]:g}',ha='center',va='center',fontsize=16)
ax.set(xticks=range(4),yticks=range(4),xticklabels=['↑↑','↑↓','↓↑','↓↓'],yticklabels=['↑↑','↑↓','↓↑','↓↓'])
ax.set_title('ρmc = P / 2：每个微观态概率 1/2',pad=15,weight='bold')
save(fig,'03-microcanonical-shell')

fig, axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ax=axes[0]
for en,label,c in [(1,'|↑〉：自旋沿 +z，磁矩沿 −z',ORANGE),(-1,'|↓〉：自旋沿 −z，磁矩沿 +z',TEAL)]:
    ax.hlines(en,0,1.1,color=c,lw=4); ax.text(0,en+.18,label,fontsize=12)
ax.annotate('',(1.25,1),(1.25,-1),arrowprops=dict(arrowstyle='<->',lw=2,color=BLUE))
ax.text(1.35,0,'Δ = 2a\n= g μB B',va='center',fontsize=13)
ax.set(ylim=(-1.6,1.7),xlim=(-.1,2.1),xticks=[],yticks=[-1,1],yticklabels=['−a','+a'],ylabel='自旋塞曼能量')
ax.set_title('电子磁矩与自旋反向：H = +a σz',loc='left',weight='bold')
ax=axes[1]
x=np.linspace(0,3,301); ax.plot(x,expit(2*x),lw=3,color=TEAL,label='低能态 p↓');ax.plot(x,expit(-2*x),lw=3,color=ORANGE,label='高能态 p↑')
ax.axvline(.6717138,color=INK,ls=':',label='B = 1 T，T = 1 K，g = 2')
ax.set(xlabel='x = a / (kB T)',ylabel='占据概率',ylim=(0,1.04));ax.legend();ax.set_title('温度降低或磁场增强 → 低能态更占优势',loc='left',weight='bold')
save(fig,'04-zeeman-populations')

fig, axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
x=np.linspace(0,5,401);t=np.tanh(x)
curves=[(t,'平均磁矩 / (g μB / 2)',TEAL),(np.logaddexp(x,-x)-x*t,'熵 S / kB',BLUE),(.5*(1+t*t),'纯度 Tr(ρ²)',BLUE),(x*x/np.cosh(x)**2,'热容 CB / kB',ORANGE)]
for ax,(y,label,c) in zip(axes.flat,curves):
    ax.plot(x,y,color=c,lw=3);ax.set(xlabel='x = a / (kB T)',ylabel=label);ax.grid(alpha=.18)
axes[0,1].axhline(np.log(2),color=INK,ls=':',lw=1);axes[1,0].set_ylim(.45,1.03)
fig.suptitle('同一个 2 × 2 密度矩阵，给出四种热平衡性质',fontsize=20,weight='bold')
save(fig,'05-thermal-observables')

fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
ax=axes[0]
box(ax,(.1,.65),.32,.2,'|0〉\nN=0，E=0',BLUE);box(ax,(.61,.65),.32,.2,'|↑〉\nN=1，E=ε0+a',ORANGE)
box(ax,(.1,.18),.32,.2,'|↓〉\nN=1，E=ε0−a',TEAL);box(ax,(.61,.18),.32,.2,'|↑↓〉\nN=2，E=2ε0',BLUE)
ax.set(xlim=(0,1.04),ylim=(0,1.02));ax.axis('off');ax.set_title('一个空间轨道，四个 Fock 基态（无库仑作用）',fontsize=14,weight='bold')
ax=axes[1]; d=np.linspace(-4,4,401);x0=.6717138
weights=np.array([np.zeros_like(d),d-x0,d+x0,2*d]);p=softmax(weights,axis=0)
for y,label,c,ls in zip(p,['P0','P↑','P↓','P↑↓'],[INK,ORANGE,TEAL,BLUE],['--','-','-','--']):ax.plot(d,y,label=label,color=c,lw=2.5,ls=ls)
ax.axvline(0,color='#aaaaaa',lw=1);ax.set(xlabel='(μ − ε0) / (kB T)',ylabel='四个多体态的概率',ylim=(0,1.03));ax.legend();ax.set_title('B = 1 T，T = 1 K；改变粒子库的化学势',fontsize=14,weight='bold')
save(fig,'06-grand-canonical')

# Independent validation: matrix exponential versus analytic spin expression;
# Fock-state enumeration versus independent Fermi occupations; partial trace.
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1,-1]).astype(complex);I=np.eye(2)
rng=np.random.default_rng(20261001);errors=[];checks=0
for x0 in [0,1e-6,.1,.6717138,1,5,20]:
    for _ in range(12):
        n=rng.normal(size=3);n/=np.linalg.norm(n);h=sum(nj*sj for nj,sj in zip(n,[sx,sy,sz]));q=expm(-x0*h);q/=np.trace(q)
        expected=(I-np.tanh(x0)*h)/2;errors.append(float(np.max(np.abs(q-expected))))
        assert np.allclose(q,expected,atol=2e-14);assert np.min(np.linalg.eigvalsh(q))>-2e-14
        assert np.allclose(q,q.conj().T);assert np.isclose(np.trace(q),1);checks+=4
for d0 in [-5,-1,0,.7,5]:
    for x0 in [0,.6717138,2]:
        p=softmax([0,d0-x0,d0+x0,2*d0]);fup=expit(d0-x0);fdn=expit(d0+x0)
        assert np.allclose(p,[(1-fup)*(1-fdn),fup*(1-fdn),(1-fup)*fdn,fup*fdn])
        ns=np.array([0,1,1,2]);mean=p@ns;var=p@(ns**2)-mean**2
        assert np.isclose(mean,fup+fdn);assert np.isclose(var,fup*(1-fup)+fdn*(1-fdn));checks+=3
        conditioned=p[1:3]/sum(p[1:3]);assert np.allclose(conditioned,softmax([-x0,x0]));checks+=1
rho_mc=np.diag([0,.5,.5,0]); reduced=np.trace(rho_mc.reshape(2,2,2,2),axis1=1,axis2=3)
assert np.allclose(reduced,I/2);assert np.isclose(np.trace(rho_mc@rho_mc),.5);checks+=2
# Use derivative checks to test the thermodynamic identities independently.
for x0 in [.01,.2,.6717138,1.2,3]:
    step=1e-5;lnz=lambda t: np.logaddexp(t,-t)
    deriv=(lnz(x0+step)-lnz(x0-step))/(2*step)
    assert np.isclose(deriv,np.tanh(x0),atol=1e-9)
    curv=(lnz(x0+step)-2*lnz(x0)+lnz(x0-step))/step**2
    assert np.isclose(curv,1/np.cosh(x0)**2,atol=1e-5);checks+=2
muB_eV_T=5.7883817982e-5;kB_eV_K=8.617333262145e-5
x0=muB_eV_T/kB_eV_K;p=softmax([-x0,x0]);pg=softmax([0,-x0,x0,0])
result={'checks_passed':checks,'max_spin_matrix_error':max(errors),'model':{'g':2,'B_T':1,'T_K':1,'orbital_and_coulomb_terms_omitted':True},'canonical':{'x':x0,'p_up':p[0],'p_down':p[1],'energy_eV':-muB_eV_T*np.tanh(x0),'entropy_over_kB':float(-p@np.log(p)),'purity':float(p@p),'moment_over_muB':float(np.tanh(x0)),'heat_capacity_over_kB':float(x0*x0/np.cosh(x0)**2)},'grand_at_mu_equals_epsilon0':{'probabilities':pg.tolist(),'mean_number':float(pg@np.array([0,1,1,2])),'number_variance':float(pg@np.array([0,1,1,4])-1)}}
Path(__file__).with_name('validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
