from common import *
import math
lc=J('length_confound')
# Table 9 (tab22) is checked in t_tables.py; below are the derived numbers quoted in Sec. 4.3 and App. B
# derived text numbers
print('--- per-unit loss (-ln kept / dlogL)')
res={}
for p in PAIRS:
    for iv in lc['pairs'][p]['intervals']:
        u={k:-math.log(iv['kept_vs_half_reference'][k])/iv['log_len_drop'] if iv['log_len_drop']>0 else float('nan') for k in STUD}
        res[(p,iv['interval'])]=u
        print(p,iv['interval'],iv['class'],{k:round(v,3) for k,v in u.items()}, 'markers logratio',round(iv['marker_log_ratio'],3) if 'marker_log_ratio' in iv else None,
              'per marker', {k:round(-math.log(iv['kept_vs_half_reference'][k])/iv['marker_log_ratio'],3) for k in STUD} if iv.get('marker_log_ratio') else '')
print('A switch/held ratio', {k: round(res[('A','0.2->0.4')][k]/res[('A','0.0->0.2')][k],3) for k in STUD})
held=[v for (p,i),u in res.items() for v in u.values() if p in 'BC' and dict((iv['interval'],iv['class']) for iv in lc['pairs'][p]['intervals'])[i]=='held']
print('B,C held per-unit range',round(min(held),3),round(max(held),3))
for p,i in [('B','0.4->0.6'),('C','0.4->0.6'),('C','0.6->0.8'),('B','0.6->0.8')]: print(p,i,{k:round(v,3) for k,v in res[(p,i)].items()})
print('-ln 0.78 of min held B/C kept:', )
for p in 'BC':
  for iv in lc['pairs'][p]['intervals']:
    if iv['class']=='held': print(p,iv['interval'],{k:round(iv['kept_vs_half_reference'][k],4) for k in STUD}, 'shrink',round(1-math.exp(-iv['log_len_drop']),3))
print('LLO region dims:', {k:v.get('region_dims') for k,v in lc['leave_lambda_out_region'].items()})
print('train tokens lambda0', {p: round(lc['pairs'][p]['tokens']['0.0']) for p in PAIRS})
print('think', {p: lc['pairs'][p]['think_rate'] for p in PAIRS})
