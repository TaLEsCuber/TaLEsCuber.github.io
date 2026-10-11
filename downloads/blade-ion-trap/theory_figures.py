"""Reproduce the article's ideal-model figures and numerical example.
Requires numpy, scipy, matplotlib. Run from any directory.
These calculations do not represent a finite-element model of an actual trap.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.constants import elementary_charge as e, atomic_mass, hbar, epsilon_0
from scipy.integrate import solve_ivp
from scipy.special import mathieu_a, mathieu_b
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'img' / 'blade-ion-trap-theory'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})
blue, red, green = '#245ca6', '#c5573b', '#24857b'

# A schematic, not the boundary of the analytic solution in the other panels.
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
ax = axs[0]
base = np.array([[0.63, 0], [1.4, -.27], [1.4, .27]])
for angle, col, label in [(0, red, 'RF'), (np.pi, red, 'RF'),
                          (np.pi/2, blue, 'RF ground / DC'),
                          (3*np.pi/2, blue, 'RF ground / DC')]:
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ax.add_patch(Polygon(base @ rot.T, facecolor=col, alpha=.85))
ax.scatter(0, 0, s=50, color='#222222', zorder=5)
ax.text(.08, .08, 'ion / z axis', fontsize=10)
ax.text(0, 1.53, 'RF ground / DC', ha='center', color=blue)
ax.text(0, -1.60, 'RF ground / DC', ha='center', color=blue)
ax.text(1.23, .39, 'RF', ha='center', color=red)
ax.text(-1.23, .39, 'RF', ha='center', color=red)
ax.set(xlim=(-1.8,1.8), ylim=(-1.8,1.8), aspect='equal', title='(a) Blade cross-section: schematic')
ax.axis('off')
grid = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(grid, grid)
for ax, Z, title in [(axs[1], X*X-Y*Y, '(b) Instantaneous RF potential'),
                     (axs[2], X*X+Y*Y, '(c) RF pseudopotential energy')]:
    ax.contourf(X, Y, Z, 20, cmap='RdBu_r' if ax is axs[1] else 'YlGnBu')
    ax.contour(X, Y, Z, 9, colors='white', linewidths=.5, alpha=.6)
    ax.set(xlabel='x / reference length', ylabel='y / reference length',
           aspect='equal', title=title)
fig.suptitle('Blade electrodes implement a quadrupole field near the center', fontsize=16)
fig.savefig(OUT / 'field-and-geometry.png', dpi=180)
plt.close(fig)

# Mathieu equation x'' + [a - 2 q cos(2 tau)] x = 0.
qgrid = np.linspace(0, 1.1, 700)
lo, hi = mathieu_a(0, qgrid), mathieu_b(1, qgrid)
qcut = brentq(lambda q: mathieu_b(1, q), .8, 1.)
a_demo, q_demo = -.003, -.25
beta_approx = np.sqrt(a_demo+q_demo*q_demo/2)
tau = np.linspace(0, 70*np.pi, 24000)
x_init = 1-q_demo/2
sol = solve_ivp(lambda t,s: [s[1], -(a_demo-2*q_demo*np.cos(2*t))*s[0]],
                [0,tau[-1]], [x_init,0], t_eval=tau, rtol=1e-10, atol=1e-12,
                max_step=.05)
fig, axs = plt.subplots(1,2,figsize=(13,4.8), constrained_layout=True)
ax = axs[0]
ax.fill_between(qgrid, lo, hi, color='#d7eee8', label='First stable band (one coordinate)')
ax.plot(qgrid, lo, color=green, label=r'$a_0(|q|)$')
ax.plot(qgrid, hi, color=blue, label=r'$b_1(|q|)$')
ax.axhline(0, color='gray', lw=.8)
ax.scatter([qcut],[0], color=red)
ax.annotate(f'{qcut:.4f}', (qcut,0), xytext=(.78,.15), arrowprops={'arrowstyle':'->'})
ax.scatter([abs(q_demo)],[a_demo], color='#222222', zorder=6)
ax.set(xlabel=r'$|q|$', ylabel='a', ylim=(-.6,1.05), xlim=(0,1.1), title='(a) Exact Mathieu stability boundaries')
ax.legend(fontsize=8,loc='upper right')
ax = axs[1]
ax.plot(tau/(np.pi), sol.y[0], color=blue, lw=.8, label='Integrated RF trajectory')
ax.plot(tau/np.pi, np.cos(beta_approx*tau), color=red, lw=1.2, ls='--', label='Lowest-order secular approximation')
ax.set(xlabel='Time / RF period', ylabel='Normalized displacement',
       title=r'(b) $a=-0.003$, $q=-0.25$')
ax.legend(fontsize=8,loc='upper right')
fig.savefig(OUT/'stability-and-motion.png', dpi=180)
plt.close(fig)

m = 172*atomic_mass
V, r0, eta, Omega = 300., 500e-6, .8, 2*np.pi*20e6
wz = 2*np.pi*200e3
q = 2*e*eta*V/(m*r0**2*Omega**2)
a = -2*wz**2/Omega**2
wrf = e*eta*V/(np.sqrt(2)*m*Omega*r0**2)
wr = np.sqrt(wrf**2-wz**2/2)
xdc = e*10/(m*wr**2)
beta = np.sqrt(a+q*q/2)
def fundamental(t,s):
    M = np.asarray(s).reshape(2,2)
    A = np.array([[0,1],[-(a+2*q*np.cos(2*t)),0]])
    return (A@M).ravel()
mono=solve_ivp(fundamental,[0,np.pi],np.eye(2).ravel(),rtol=1e-12,atol=1e-13,max_step=.01).y[:,-1].reshape(2,2)
beta_exact = np.arccos(np.trace(mono)/2)/np.pi
report = {'mass_kg':m,'q_magnitude':q,'a_radial':a,'f_rf_only_Hz':wrf/(2*np.pi),
          'f_radial_pseudo_Hz':wr/(2*np.pi),'f_radial_floquet_Hz':beta_exact*Omega/(4*np.pi),
          'floquet_vs_pseudo_relative':beta_exact/beta-1,
          'displacement_at_10Vm_nm':xdc*1e9,'excess_micromotion_nm':q*xdc/2*1e9,
          'ground_state_spread_nm':np.sqrt(hbar/(2*m*wr))*1e9,
          'two_ion_spacing_um':(2*e**2/(4*np.pi*epsilon_0*m*wz**2))**(1/3)*1e6,
          'q_cutoff_at_a0':qcut,'monodromy_determinant':np.linalg.det(mono)}
assert abs(np.linalg.det(mono)-1)<1e-9
assert .9<qcut<.91
assert 0<report['floquet_vs_pseudo_relative']<.02
(Path(__file__).parent/'numerical-example.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
