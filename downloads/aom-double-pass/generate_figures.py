"""Original illustrations for the AOM double-pass article. No measured data.

Run: python source/downloads/aom-double-pass/generate_figures.py
Requires numpy, matplotlib, Pillow. Outputs SVG + PNG in source/img/aom-double-pass.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib import font_manager
from PIL import Image, ImageOps, ImageDraw

SOURCE = Path(__file__).resolve().parents[2]
OUT = SOURCE / 'img' / 'aom-double-pass'
OUT.mkdir(parents=True, exist_ok=True)
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'mathtext.fontset': 'dejavusans'})
INK, BLUE, ORANGE, GREEN, GRAY = '#182e49', '#2469c8', '#d46b24', '#168879', '#6b7c93'
BG = '#f5f8fc'
files = []

def canvas(title, sub, height=6):
    fig, ax = plt.subplots(figsize=(12, height), facecolor=BG)
    ax.set(xlim=(0, 12), ylim=(0, height))
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=.96)
    ax.text(.25, height-.45, title, fontsize=21, weight='bold', color=INK)
    ax.text(.25, height-.86, sub, fontsize=11, color=GRAY)
    return fig, ax

def arrow(ax, a, b, color=BLUE, lw=2.5, style='->'):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def box(ax, x, y, w, h, label, color=BLUE, size=12):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor=color, lw=1.8, zorder=3))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', color=INK, fontsize=size, zorder=4)

def save(fig, name):
    fig.savefig(OUT / (name+'.svg'), facecolor=BG)
    fig.savefig(OUT / (name+'.png'), dpi=180, facecolor=BG)
    files.append(OUT / (name+'.png'))
    plt.close(fig)

fig, ax = canvas('02  两次衍射，频率逐次相加', '频率符号按实际升频 / 降频定义；光束向左返回不代表频移变负。')
for y, lab, vals, color in [(3.8, '升频双程', ['ν₀', 'ν₀ + fRF', 'ν₀ + 2fRF'], BLUE),
                            (2.3, '降频双程', ['ν₀', 'ν₀ − fRF', 'ν₀ − 2fRF'], ORANGE)]:
    ax.text(.35,y+.18,lab,color=color,weight='bold',fontsize=14)
    for x, val in zip([2.1,5.5,8.9],vals):
        box(ax,x,y-.15,2.35,.72,val,color,15)
    arrow(ax,(4.5,y+.2),(5.4,y+.2),color)
    arrow(ax,(7.9,y+.2),(8.8,y+.2),color)
ax.text(3.3,4.7,'入射光',ha='center',color=GRAY)
ax.text(6.7,4.7,'第一次衍射后',ha='center',color=GRAY)
ax.text(10.1,4.7,'第二次衍射后',ha='center',color=GRAY)
ax.text(.45,1.25,'例：RF = 80 MHz → 升频双程输出相对输入高 160 MHz',fontsize=17,color=INK)
ax.text(.45,.55,'只有目标的两次一级衍射光满足此频率链；零级和杂散路径需另外追踪。',color=GRAY)
save(fig,'01-frequency')

fig, ax = canvas('01  PBS + QWP + cat’s-eye 的双程光路', '结构示意，不按比例；蓝色为去程，橙色虚线为回程（为辨认而略微错开）。', 7)
y=3.8
arrow(ax,(.4,y),(3.95,y))
box(ax,1.65,3.5,.7,.65,'PBS',size=11)
box(ax,4,3.4,.65,.85,'AOM',size=10)
ax.text(.45,4.15,'输入 ν₀',color=BLUE)
ax.plot([4.65,7.45,10.6],[y,4.8,4.8],color=BLUE,lw=2.5)
arrow(ax,(8.15,4.8),(9.25,4.8))
ax.plot([10.6,7.45,4.5,2],[4.69,4.69,3.66,3.66],color=ORANGE,lw=2,ls='--')
arrow(ax,(6.8,4.45),(5.6,4.03),ORANGE)
arrow(ax,(3.7,3.66),(2.55,3.66),ORANGE)
arrow(ax,(2,3.65),(2,1.45),ORANGE)
box(ax,1.05,.8,1.9,.65,'输出 / 光纤',ORANGE)
ax.text(2.35,2.2,'ν₀ + 2fRF',color=ORANGE,fontsize=14)
box(ax,5.65,3.85,.28,1.0,'',GREEN)
ax.text(5.79,5.08,'QWP',ha='center',color=GREEN)
ax.add_patch(Ellipse((7.45,y),.3,2.4,fc='#dceafe',ec=BLUE,lw=2,zorder=3))
ax.text(7.45,5.45,'透镜 L',ha='center',color=INK)
ax.plot([10.6,10.6],[3.35,5.3],color=INK,lw=5)
ax.text(10.6,5.55,'平面镜 M',ha='center',color=INK)
ax.plot([4.65,9.25],[3.8,3.8],color=GRAY,lw=1.5,ls=':')
box(ax,8.9,3.57,.5,.45,'挡',GRAY,10)
ax.text(8.1,3.05,'零级光 → 光束挡',color=GRAY)
ax.text(8.2,5.12,'ν₀ + fRF',color=BLUE)
arrow(ax,(4.32,2.6),(7.45,2.6),GRAY,1.2,'<->')
arrow(ax,(7.45,2.6),(10.6,2.6),GRAY,1.2,'<->')
ax.text(5.88,2.15,'d ≈ fL',ha='center',color=GRAY)
ax.text(9.02,2.15,'s ≈ fL',ha='center',color=GRAY)
ax.text(4,.85,'回程经 QWP 后线偏振转过 90°，由 PBS 导向输出口。\n前提：AOM 支持这两种偏振的所需衍射过程。',color=INK,linespacing=1.7)
save(fig,'02-layout')

fig, ax = canvas('03  猫眼怎样把扫角变成镜面上的平移？', '仅画三种 RF 频率的主光线；透镜前的共同出发点代表 AOM 相互作用区。',6.6)
x0,xL,xM,y0=1.6,6.1,10.5,2.5
ax.plot([.7,11.1],[y0,y0],color=GRAY,ls=':',lw=1)
ax.add_patch(Ellipse((xL,y0),.3,3.0,fc='#dceafe',ec=BLUE,lw=2,zorder=3))
ax.plot([xM,xM],[1.9,4.65],color=INK,lw=5)
for h,c,lab in [(3.0,GREEN,'低 RF'),(3.4,BLUE,'中心 RF'),(3.8,ORANGE,'高 RF')]:
    ax.plot([x0,xL,xM],[y0,h,h],color=c,lw=2.3)
    arrow(ax,(7.0,h),(8.0,h),c)
    arrow(ax,(9.6,h-.09),(8.6,h-.09),c,1.5)
    ax.text(10.75,h,lab,color=c,va='center',fontsize=11)
ax.scatter([x0],[y0],s=70,color=INK,zorder=6)
ax.text(x0,2.02,'AOM',ha='center',color=INK)
ax.text(xL,4.3,'L',ha='center',color=INK)
ax.text(xM,4.95,'M',ha='center',color=INK)
ax.text(2.0,4.55,'主光线到达透镜时：\nh ≈ fL · θ\n出射角 θ − h/fL ≈ 0',color=INK,linespacing=1.65)
arrow(ax,(x0,.9),(xL,.9),GRAY,1.1,'<->')
arrow(ax,(xL,.9),(xM,.9),GRAY,1.1,'<->')
ax.text((x0+xL)/2,.57,'fL',ha='center',color=GRAY)
ax.text((xL+xM)/2,.57,'fL',ha='center',color=GRAY)
ax.text(.35,.2,'理想回程主光线沿原路返回；有限光束的波前复原还要求正确的镜距。',color=GRAY)
save(fig,'03-cat-eye')

fig, ax = canvas('04  偏振负责把返回光从输入光中分离', '下图画的是各位置的横向电场轨迹；不以圆偏振左右手性命名，避免观察方向歧义。',5.4)
centers=[1.2,3.6,6,8.4,10.8]
labels=['入射线偏振','第一次过 QWP','镜面反射','第二次过 QWP','PBS 反射输出']
for i,(x,label) in enumerate(zip(centers,labels)):
    ax.add_patch(Circle((x,2.75),.58,fc='white',ec='#d9e1ec',lw=1.5))
    if i==0:
        arrow(ax,(x-.43,2.75),(x+.43,2.75),BLUE,2.5,'<->')
    elif i in [1,2]:
        t=np.linspace(0,2*np.pi,160)
        ax.plot(x+.4*np.cos(t),2.75+.4*np.sin(t),color=GREEN,lw=2.5)
    else:
        arrow(ax,(x,2.32),(x,3.18),ORANGE,2.5,'<->')
    ax.text(x,1.7,label,ha='center',fontsize=11,color=INK)
    if i<4: arrow(ax,(x+.72,2.75),(x+1.65,2.75),GRAY,1.5)
ax.text(.45,.85,'QWP 快轴与入射线偏振约成 45°；往返等效半波延迟。',fontsize=15,color=INK)
ax.text(.45,.25,'图示假定 AOM 保持所示偏振类型，且反射镜不额外引入显著偏振畸变。',color=GRAY)
save(fig,'04-polarization')

fig, axes=plt.subplots(1,2,figsize=(12,5.5),facecolor=BG)
fig.subplots_adjust(left=.075,right=.965,bottom=.2,top=.74,wspace=.28)
fig.text(.045,.91,'05  RF 带宽与光学扫频范围必须分开读',fontsize=21,weight='bold',color=INK)
fig.text(.045,.83,'假设模型：单程峰值 0.85，RF FWHM = 40 MHz；两程效率相同，忽略额外损耗。',color=GRAY,fontsize=11)
d=np.linspace(-55,55,600)
eta=.85*np.exp(-4*np.log(2)*(d/40)**2)
axes[0].plot(d,eta,color=BLUE,lw=2.5,label='单程 η')
axes[0].plot(d,eta**2,color=ORANGE,lw=2.5,label='双程 η²')
axes[0].set(xlabel='RF 相对中心的偏移 / MHz',ylabel='衍射效率',ylim=(0,1),title='在同一 RF 横轴上：双程更窄')
axes[1].plot(d,eta/.85,color=BLUE,lw=2.5,label='单程：40 MHz')
axes[1].plot(2*d,(eta/.85)**2,color=ORANGE,lw=2.5,label='双程：56.6 MHz')
axes[1].axhline(.5,color=GRAY,ls='--',lw=1)
axes[1].set(xlabel='光频相对各自中心的偏移 / MHz',ylabel='按各自峰值归一化的功率',ylim=(0,1.08),xlim=(-65,65),title='换成光频横轴：还要乘以 2')
for ax in axes:
    ax.set_facecolor('white'); ax.grid(alpha=.2); ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
fig.text(.075,.05,'图为公式计算，非器件实测。RF 双程 FWHM = 28.3 MHz；光频双程 FWHM = 56.6 MHz。',color=GRAY,fontsize=11)
save(fig,'05-bandwidth')

fig, axes=plt.subplots(1,2,figsize=(12,5.5),facecolor=BG)
fig.subplots_adjust(left=.09,right=.965,bottom=.2,top=.74,wspace=.33)
fig.text(.045,.91,'06  小光束切换更快，但发散角也更大',fontsize=21,weight='bold',color=INK)
fig.text(.045,.83,'示例：声速 vs = 4.0 km/s，真空波长 780 nm，M² = 1；并非特定 AOM 的参数。',color=GRAY,fontsize=11)
dmm=np.linspace(.08,1.2,250)
tr=.65*(dmm*1e-3)/4000*1e9
div=780e-9/(np.pi*dmm*1e-3/2)*1e3
axes[0].plot(dmm,tr,color=BLUE,lw=2.5)
axes[0].set(xlabel='强度 1/e² 直径 d / mm',ylabel='估算 10–90% 上升时间 / ns',title='声波扫过光束的时间')
axes[1].plot(dmm,div,color=ORANGE,lw=2.5)
axes[1].set(xlabel='强度 1/e² 腰斑直径 d / mm',ylabel='高斯光束远场半发散角 / mrad',title='自由空间高斯光束的角谱宽度')
for ax in axes:
    ax.set_facecolor('white');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.text(.09,.045,'实际速度还受驱动器、声传播延迟和探测链路限制；发散角需与器件角接受度比较。',color=GRAY,fontsize=11)
save(fig,'06-speed-tradeoff')

thumbs=[]
for path in files:
    im=Image.open(path).convert('RGB')
    im.thumbnail((840,430))
    cell=Image.new('RGB',(860,470),'white')
    cell.paste(im,((860-im.width)//2,10))
    ImageDraw.Draw(cell).text((15,445),path.stem,fill='black')
    thumbs.append(cell)
sheet=Image.new('RGB',(1720,1410),'#dce3ed')
for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*860,(i//2)*470))
qa=SOURCE.parent/'tmp'/'aom-double-pass'
qa.mkdir(parents=True,exist_ok=True)
sheet.save(qa/'contact-sheet.png')
print(f'Generated {len(files)} original figures in {OUT}')
