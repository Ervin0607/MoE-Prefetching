"""Recompute native energy for unknown link/predictor costs and background range."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    df=pd.read_csv(a.directory/'metrics.csv'); rows=[]
    for bg in [70,127,200]:
        for link in [0,10,100]:
            for predictor in [0,1,5]:
                for _,r in df.iterrows():
                    pred_s=r.tokens*31*1e-6 if r['mode'] not in ('cache','streaming') and r.budget!=0 else 0
                    e=r.gpu_dynamic_j+r.host_read_j+r.hbm_write_j+r.host_bytes*link*1e-12+r.elapsed_s*bg+pred_s*predictor
                    rows.append(dict(mode=r['mode'],budget=r.budget,sequence_id=r.sequence_id,cache_bytes=r.cache_bytes,link_gbps=r.link_gbps,
                        background_w=bg,link_pj_byte=link,predictor_w=predictor,total_j=e,joules_per_token=e/r.tokens))
    out=pd.DataFrame(rows)
    keys=['sequence_id','cache_bytes','link_gbps','background_w','link_pj_byte','predictor_w']
    controls=out[out['mode']=='cache'][keys+['total_j']].rename(columns={'total_j':'cache_j'})
    out=out.merge(controls,on=keys,how='left',validate='many_to_one')
    out['saving_vs_cache']=1-out.total_j/out.cache_j
    out.to_csv(a.directory/'energy_sensitivity.csv',index=False)
    print('Wrote',len(out),'energy scenarios')
if __name__=='__main__':main()
