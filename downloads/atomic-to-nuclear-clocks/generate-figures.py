"""Reproduce teaching figures and numerical checks for the atomic/nuclear clock article.

Requirements: Python 3, NumPy, Matplotlib. Run from any directory.
Curves are analytic illustrations, not digitized experimental observations.
Paper parameters: Nature DOI 10.1038/s41586-026-11122-1 and 11084-4.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import fontManager

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'source/img/atomic-to-nuclear-clocks'
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    fontManager.addfont(str(FONT))
plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
    'font.sans-serif': ['Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
    'font.size': 16, 'axes.unicode_minus': False,
    'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc',
    'text.color': '#172b43', 'axes.labelcolor': '#172b43',
    'xtick.color': '#475569', 'ytick.color': '#475569',
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none',
})
INK, TEAL, BLUE, GOLD = '#172b43', '#087f8c', '#355cbb', '#b77820'

def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)

def canvas(title, subtitle, size=(14, 7)):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.02, .975, title, fontsize=25, fontweight='bold', va='top')
    ax.text(.02, .89, subtitle, fontsize=15, color='#52657b', va='top')
    return fig, ax

def card(ax, x, y, w, h, title, lines, color=TEAL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=.018',
                              facecolor='white', edgecolor=color, linewidth=1.8))
    ax.text(x+w/2, y+h*.74, title, ha='center', va='center', fontsize=18,
            color=color, fontweight='bold')
    ax.text(x+w/2, y+h*.34, lines, ha='center', va='center', fontsize=15, linespacing=1.6)

def arrow(ax, start, end, color=INK, style='-|>'):
    ax.annotate('', end, start, arrowprops={'arrowstyle': style, 'lw': 2.1,
                                         'color': color, 'mutation_scale': 17})

# Cover: three reference families. All gaps have arbitrary graphical scale.
fig, ax = canvas('从原子钟到核钟', '共同任务：把量子跃迁变成稳定、可复现的频率参考', (14, 7.4))
for x, name, species, freq, desc, color in [
    (.035, '微波原子钟', '¹³³Cs', '9.192 631 770 GHz', '基态超精细跃迁', BLUE),
    (.365, '光学原子钟', '⁸⁷Sr 等', '数百 THz', '电子能级之间的窄跃迁', TEAL),
    (.695, '钍-229 核钟', '²²⁹Th', '约 2.02 PHz', '核基态与低能异构态', GOLD),
]:
    ax.add_patch(FancyBboxPatch((x, .16), .27, .63, boxstyle='round,pad=.012',
                              facecolor='white', edgecolor='#d4dce8'))
    ax.text(x+.135, .73, name, fontsize=19, color=color, ha='center', weight='bold')
    ax.text(x+.135, .65, species, fontsize=19, ha='center')
    for y in [.36, .53]:
        ax.plot([x+.055, x+.215], [y, y], color=color, lw=3)
    arrow(ax, (x+.135, .37), (x+.135, .52), color)
    ax.text(x+.135, .275, freq, ha='center', fontsize=17, weight='bold', color=color)
    ax.text(x+.135, .205, desc, ha='center', fontsize=14)
ax.text(.5, .07, '能级间距为示意  ·  核跃迁不改变元素种类  ·  参考频率高不等于所有误差更小',
        ha='center', fontsize=15)
save(fig, '00-cover')

fig, ax = canvas('光学钟如何形成反馈闭环', '快环维持短期相干；量子共振校正慢漂移；光频梳连接输出频率。', (14, 8))
card(ax, .04, .60, .22, .18, '超稳光学腔', '短期稳定性', BLUE)
card(ax, .385, .60, .22, .18, '钟激光', '受控本地振荡器', TEAL)
card(ax, .735, .60, .22, .18, '原子 / 原子核', '量子跃迁参考', GOLD)
card(ax, .735, .27, .22, .18, '探测与鉴频', '布居 / 吸收 → 误差', GOLD)
card(ax, .385, .27, .22, .18, '反馈执行器', '调整频率偏置', TEAL)
card(ax, .04, .27, .22, .18, '光频梳与计数', '光频 → 射频 → 时间', BLUE)
arrow(ax, (.28, .70), (.37, .70), BLUE)
ax.text(.323, .75, '快环', ha='center', fontsize=14)
arrow(ax, (.62, .70), (.72, .70))
ax.text(.67, .75, '探测', ha='center', fontsize=14)
arrow(ax, (.845, .58), (.845, .47))
arrow(ax, (.72, .36), (.62, .36), GOLD)
arrow(ax, (.495, .47), (.495, .58), TEAL)
ax.text(.505, .52, '校正', fontsize=14, va='center')
ax.plot([.40, .31, .31, .15], [.60, .53, .51, .51], color=BLUE, lw=2)
arrow(ax, (.15, .51), (.15, .47), BLUE)
ax.text(.5, .11, '锁腔与锁原子承担不同任务；实际反馈可施加在激光、AOM、EOM 或频梳支路。',
        ha='center', fontsize=15)
save(fig, '01-feedback')

def ramsey(delta, T, contrast=1):
    return .5*(1+contrast*np.cos(2*np.pi*np.asarray(delta)*T))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout='constrained')
d = np.linspace(-3.1, 3.1, 1400)
for T, c in [(.25, BLUE), (.5, TEAL)]:
    axes[0].plot(d, ramsey(d, T), color=c, lw=2.5, label=f'T = {T:g} s')
axes[0].set(xlabel='失谐 δ / Hz', ylabel='激发态概率 Pₑ', title='相干演化越长，中央条纹越窄', ylim=(-.04, 1.05))
axes[0].legend(loc='lower center', fontsize=13)
dd = np.linspace(-.9, .9, 600)
T, step = .5, .5
err = ramsey(dd+step, T)-ramsey(dd-step, T)
axes[1].plot(dd, err, color=GOLD, lw=2.7)
axes[1].plot(dd, -2*np.pi*T*dd, color='#8193a8', lw=1.5, ls='--', label='中心附近的线性近似')
axes[1].set(xlabel='中心失谐 δ / Hz', ylabel='双点差分信号 e', title='误差信号给出频率偏差方向', ylim=(-1.12, 1.12))
axes[1].axhline(0, lw=1, color='#a5b4c5')
axes[1].axvline(0, lw=1, color='#a5b4c5')
axes[1].legend(fontsize=13, loc='lower left')
for a in axes:
    a.grid(alpha=.15)
fig.suptitle('理想 Ramsey 光谱示意：无有限脉冲包络与退相干', fontsize=20)
save(fig, '02-ramsey')

fig, ax = canvas('两种连续波钍-229 核钟路线', '均利用核吸收信号校正振荡器；晶体中的实际线宽远大于自然线宽。', (15, 8))
ax.text(.025, .79, 'Huang 等：镉蒸气四波混频', color=TEAL, weight='bold', fontsize=19)
ax.text(.025, .405, 'Toscani De Col 等：逐级倍频', color=BLUE, weight='bold', fontsize=19)
xs = [.025, .28, .535, .79]
for y, color, cards in [
    (.535, TEAL, [('750 / 710 nm', '光频梳传递相干性'), ('镉蒸气混频', '375 + 375 + 710 nm'),
                 ('148.4 nm', '照射 Th:CaF₂'), ('光电管鉴频', '反馈 → AOM')]),
    (.15, BLUE, [('1187 nm', '钟激光锁高精细腔'), ('连续倍频', '594 → 297 → 148 nm'),
                ('148 nm', '照射 Th:CaF₂'), ('PMT 鉴频', '反馈 → EOM')]),
]:
    for x, (title, body) in zip(xs, cards):
        card(ax, x, y, .18, .18, title, body, color)
    for x in xs[:-1]:
        arrow(ax, (x+.193, y+.09), (x+.24, y+.09), color)
ax.text(.50, .49, 'VUV 产生约 10 μW；优化配置探测到 350 nW；钟线宽约 27 kHz', ha='center', fontsize=14)
ax.text(.50, .105, '到达 PMT 的 VUV 约 65 pW；钟线宽约 100 kHz；外部 Yb⁺ 钟用于比较', ha='center', fontsize=14)
ax.text(.50, .035, '依据 Nature (2026) 两篇论文重绘的功能示意；功率注明所在位置，不能混用。', ha='center', fontsize=14, color='#52657b')
save(fig, '03-nuclear-routes')

fig, ax = plt.subplots(figsize=(12, 7.2), layout='constrained')
tau_h = np.geomspace(11.8, 6000, 250)
tau_t = np.geomspace(20, 86400, 300)
ax.loglog(tau_h, 5e-13/np.sqrt(tau_h), color=TEAL, lw=3,
          label=r'Huang: $5\times10^{-13}/\sqrt{\tau/\mathrm{s}}$')
ax.loglog(tau_t, 3e-12/np.sqrt(tau_t), color=BLUE, lw=3,
          label=r'Toscani De Col: $3\times10^{-12}/\sqrt{\tau/\mathrm{s}}$')
ax.scatter([86400], [3e-12/np.sqrt(86400)], s=65, color=BLUE)
ax.annotate('一天：约 1.0 × 10⁻¹⁴\n由标度公式计算', (86400, 3e-12/np.sqrt(86400)),
            (900, 3.0e-13), fontsize=15, arrowprops={'arrowstyle': '->', 'color': BLUE})
ax.set(xlim=(10, 1e5), ylim=(3e-15, 1e-12), xlabel='平均时间 τ / s', ylabel='分数频率不稳定度 σᵧ(τ)',
       title='论文报告的白频率噪声标度：示意曲线')
ax.grid(which='both', alpha=.18)
ax.legend(loc='lower left', fontsize=14)
fig.text(.55, .89, '非原始数据 · 非系统不确定度 · 不可无限外推', ha='center', fontsize=14, color='#52657b')
save(fig, '04-stability')

# Independent numerical sanity checks for the equations and reported scale factors.
h, c, nu = 6.62607015e-34, 299792458., 2.0204e15
Ndot = .1*350e-9/(h*nu)
shot = 27000/(1.05*nu*2.9e-4)*np.sqrt(h*nu/(.1*350e-9))*np.sqrt(11.8/10)
checks = []
for T in [.1, .25, .5, 1.]:
    assert abs(ramsey(1/(4*T), T)-.5) < 1e-12
    step = 1/(4*T)
    eps = 1e-6
    eplus = ramsey(eps+step, T)-ramsey(eps-step, T)
    eminus = ramsey(-eps+step, T)-ramsey(-eps-step, T)
    deriv = (eplus-eminus)/(2*eps)
    assert np.isclose(deriv, -2*np.pi*T, rtol=1e-8)
    checks.append({'T_s': T, 'half_height_Hz': 1/(4*T), 'error_slope_per_Hz': float(deriv)})
assert abs(shot/2.9e-13-1) < .02
report = {
    'scope': 'Analytic teaching figures and numerical checks; not experimental data or a full clock simulation.',
    'ramsey_checks': checks,
    'huang_shot_noise_coefficient': float(shot),
    'detected_photoelectron_rate_per_s': float(Ndot),
    'huang_model_at_1000_s': float(5e-13/np.sqrt(1000)),
    'toscani_model_at_one_day': float(3e-12/np.sqrt(86400)),
    'lifetime_only_linewidth_for_600_s_Hz': float(1/(2*np.pi*600)),
    'gravitational_redshift_for_1_cm': float(9.81*.01/c**2),
    'inter_crystal_fractional_difference': float(560/nu),
    'figures': sorted(p.name for p in OUT.glob('*.png')),
    'all_checks_passed': True,
}
Path(__file__).with_name('validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
