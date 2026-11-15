"""Generate report figures from archived results; timeline is illustrative."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
COLORS={'streaming':'#718096','cache':'#2878a0','combined':'#dc862f','oracle':'#368477'}
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def finish(fig,name):
    fig.savefig(OUT/name,dpi=200,bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    def bars(labels,values,colors_,title,unit,name,decimals=1):
        fig,ax=plt.subplots(figsize=(9,4.25),layout='constrained')
        positions=np.arange(len(labels))
        rects=ax.barh(positions,values,color=colors_,height=.56)
        ax.bar_label(rects,labels=[f'{v:,.{decimals}f}' for v in values],padding=7,fontsize=11)
        ax.set(yticks=positions,yticklabels=labels,xlabel=unit,title=title,xlim=(0,max(values)*1.18))
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=9)
        for tick,color in zip(ax.get_yticklabels(),colors_):
            if color in ('#dc862f','#2878a0','#368477'):tick.set_fontweight('bold')
        finish(fig,name)
    rows=read(HERE/'results/prediction_metrics.csv')
    labels=['Popular | budget 2','Temporal | budget 2','Spatial | 2, fixed table','Spatial | 2, online','Combined | 2, online','Combined | 4, online','Combined | threshold','Random | 2 (expected)']
    shades=['#a2acb6']*8;shades[4]='#dc862f';shades[7]='#d8dde2'
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
    positions=np.arange(len(labels))
    for ax,key,title in zip(axes,['recall','precision'],['Recall (%)','Precision (%)']):
        values=[float(r[key])*100 for r in rows]+[25.0]
        ax.barh(positions,values,color=shades,height=.57)
        ax.set(title=title,xlim=(0,70),xticks=[0,20,40,60],yticks=positions)
        ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
    axes[0].set_yticklabels(['Popular, b=2','Temporal, b=2','Spatial, b=2, fixed','Spatial, b=2, online','Combined, b=2, online','Combined, b=4, online','Combined, threshold','Random, b=2 (expected)'])
    axes[0].invert_yaxis();axes[0].get_yticklabels()[4].set_fontweight('bold')
    fig.supxlabel('Higher is better; b = maximum prediction budget',fontsize=11)
    finish(fig,'prediction_comparison.png')
    groups=defaultdict(list)
    for r in read(HERE/'results/native_full_new/metrics.csv'):groups[r['mode']].append(r)
    def total(m,k):return sum(float(r[k]) for r in groups[m])
    def per(m,k,s=1):return total(m,k)/total(m,'tokens')*s
    modes=['streaming','cache','temporal','combined','spatial','random','oracle']
    labels=['Streaming','Cache-only','Temporal','Combined','Spatial','Random','Oracle']
    shades=[COLORS.get(m,'#a2acb6') for m in modes]
    fig,axes=plt.subplots(1,3,figsize=(10.6,4.0),sharey=True,layout='constrained')
    from matplotlib.ticker import MaxNLocator
    panels=[([per(m,'elapsed_s',1000) for m in modes],'Latency','ms/token'),
            ([total(m,'exposed_stall_s') for m in modes],'Exposed stall','Total seconds'),
            ([total(m,'host_bytes')/1e12 for m in modes],'Host-to-GPU traffic','Total TB')]
    for ax,(values,title,unit) in zip(axes,panels):
        ax.barh(np.arange(len(labels)),values,color=shades,height=.57)
        ax.set(title=title,xlabel=unit,xlim=(0,max(values)*1.10),yticks=np.arange(len(labels)))
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    axes[0].set_yticklabels(labels);axes[0].invert_yaxis()
    for i in [1,3,6]:axes[0].get_yticklabels()[i].set_fontweight('bold')
    fig.suptitle('Same 16,384-token workload per policy - lower is better',fontsize=12)
    finish(fig,'performance_comparison.png')
    bars(labels,[per(m,'total_j') for m in modes],shades,'Estimated energy per token','Joules per token - lower is better','policy_energy.png',2)
    fig,axes=plt.subplots(2,1,figsize=(9,4.2),layout='constrained')
    for ax,duration,title in [(axes[0],3,'Useful prefetch: weights ready when requested'),(axes[1],7,'Late prefetch: remaining transfer causes a wait')]:
        ax.broken_barh([(0,5)],(1.1,.55),facecolors='#2878a0');ax.text(2.5,1.38,'Current-layer computation',ha='center',va='center',color='white',fontsize=10)
        ax.broken_barh([(0,duration)],(.2,.55),facecolors='#dc862f');ax.text(duration/2,.48,'Next-layer weight copy',ha='center',va='center',fontsize=10)
        start=max(5,duration)
        if duration>5:ax.broken_barh([(5,duration-5)],(1.1,.55),facecolors='#c85450');ax.text(6,1.38,'WAIT',ha='center',va='center',color='white')
        ax.broken_barh([(start,2)],(1.1,.55),facecolors='#368477');ax.text(start+1,1.38,'Next layer',ha='center',va='center',color='white',fontsize=9)
        ax.axvline(5,color='#444',ls='--');ax.text(5,.02,'Weights needed',ha='center',va='top',fontsize=9)
        ax.set(xlim=(0,9.5),ylim=(-.4,1.85),yticks=[],xticks=[],title=title);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False)
    fig.supxlabel('Time proceeds left to right - illustrative only, not measured durations',fontsize=10)
    finish(fig,'prefetch_timeline.png')
    print('Generated side-by-side prediction and performance panels, energy chart and timeline')
if __name__=='__main__':main()
