from common import *
import math
lc=J('length_confound')
c=Checker('Table 8 (tab22, tab:length)')
def lamkey(x):
    x=x.strip().replace('$\\to$','->')
    a,b=x.split('->'); f=lambda v: {'0':'0.0','.2':'0.2','.4':'0.4','.6':'0.6','.8':'0.8','1':'1.0'}[v]
    return f'{f(a)}->{f(b)}'
cls={'Held':'held','Switch':'switch','After':'after'}
def blocks(i):
    cur=None; n=0
    for r in rows('tab22.tex',i):
        if r[0].strip(): cur=r[0].strip(); n=0
        iv=lc['pairs'][P(cur)]['intervals'][n] if P(cur) in lc['pairs'] else lc['pairs'][cur]['intervals'][n]; n+=1
        assert iv['interval']==lamkey(r[1]), (iv['interval'],r[1])
        yield cur,iv,r
for cur,iv,r in blocks(0):   # Pair | lambda | class | dlogL | think | kept vs half reference (Q L O)
    lam=iv['interval']
    if cls[r[2].strip()]!=iv['class']: c.err.append(f'{cur} {lam} class {r[2]} vs {iv["class"]}')
    c.chk(f'{cur} {lam} dlogL',tok(r[3])[0][0],2,iv['log_len_drop'],'intervals.log_len_drop')
    t=tok(r[4]); c.chk(f'{cur} {lam} think0',t[0][0],0,iv['think'][0],'think[0]'); c.chk(f'{cur} {lam} think1',t[1][0],0,iv['think'][1],'think[1]')
    for s_,k in enumerate(STUD): c.chk(f'{cur} {lam} kept {k}',tok(r[5+s_])[0][0],2,iv['kept_vs_half_reference'][k],'kept_vs_half_reference')
for cur,iv,r in blocks(1):   # Pair | lambda | coverage LOO (Q L O) | leave-lambda-out (Q L O) | R (Q L O)
    lam=iv['interval']
    for j,fld in enumerate(['coverage_change','leave_lambda_out_coverage_change','R_change']):
        for s_,k in enumerate(STUD): c.chk(f'{cur} {lam} {fld} {k}',tok(r[2+3*j+s_])[0][0],2,iv[fld][k],f'intervals[].{fld}.{k}')
c.report()
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
