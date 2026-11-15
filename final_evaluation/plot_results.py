"""Plot aggregate outcomes; per-sequence metrics remain available in the CSV."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/moe-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    df=pd.read_csv(a.directory/'metrics.csv')
    keys=['mode','budget','link_gbps','cache_bytes']
    summary=df.groupby(keys).agg(tokens=('tokens','sum'),seconds=('elapsed_s','sum'),joules=('total_j','sum'),stall=('exposed_stall_s','sum'),bytes=('host_bytes','sum')).reset_index()
    summary['ms_per_token']=summary.seconds/summary.tokens*1000
    summary['joules_per_token']=summary.joules/summary.tokens
    summary.to_csv(a.directory/'summary.csv',index=False)
    for (bw,cap),g in summary.groupby(['link_gbps','cache_bytes']):
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        labels=g['mode']+' b='+g.budget.astype(str)
        for ax,col,title in zip(axes,['ms_per_token','joules_per_token','bytes'],['Latency (ms/token)','Energy (J/token; assumed coefficients)','Host traffic (bytes)']):
            ax.barh(labels,g[col]);ax.set_title(title)
        fig.suptitle(f'Cold-start C4 replay, batch 1: {bw:g} GB/s, {cap/1e9:g} GB cache')
        fig.tight_layout();fig.savefig(a.directory/f'comparison_{bw:g}_{cap/1e9:g}.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
