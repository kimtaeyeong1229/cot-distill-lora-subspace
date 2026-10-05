from common import *
import numpy as np
tr=J('think_rate'); td=J('teacher_density'); ss=J('seed_spread')
LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
# ---- tab01
c=Checker('tab01')
for p,s,r in cellrows('tab01.tex'):
    vals=[tok(x) for x in r[2:]]
    if s.startswith('Teacher think'):
        for i,l in enumerate(LAM): c.chk(f'{p} think {l}',vals[i][0][0],0,100*tr[P(p)]['rates'][l],'think_rate')
    elif s.startswith('Teacher markers'):
        for i,l in enumerate(LAM): c.chk(f'{p} tmark {l}',vals[i][0][0],1,td[p]['teacher'][l],'teacher_density')
    else:
        k=sname(s); cell=ss[f'{p}/{k}']['table1']
        for i,l in enumerate(LAM):
            c.chk(f'{p}/{k} {l} mean',vals[i][0][0],vals[i][0][1],cell[l]['mean'],'seed_spread.table1')
            c.chk(f'{p}/{k} {l} sd',vals[i][1][0],vals[i][1][1],cell[l]['sd'],'seed_spread.table1')
        c.chk(f'{p}/{k} r',vals[6][0][0],2,td[p]['corr'][k],'teacher_density.corr')
c.chk('caption pooled r',0.86,2,td['pooled_corr'],'pooled_corr')
c.report()
# ---- tab02
c=Checker('tab02')
for p,s,r in cellrows('tab02.tex'):
    k=sname(s); t2=ss[f'{p}/{k}']['table2']; v=[tok(x) for x in r[2:]]
    for j,(key,sc) in enumerate([('tokens0',1),('tokens0',1),('tokens1',1),('tokens1',1)]): pass
    c.chk(f'{p}/{k} tok0',v[0][0][0],0,t2['tokens0']['mean'],'table2.tokens0'); c.chk(f'{p}/{k} tok0 sd',v[0][1][0],0,t2['tokens0']['sd'],'')
    c.chk(f'{p}/{k} tok1',v[0][2][0],0,t2['tokens1']['mean'],'table2.tokens1'); c.chk(f'{p}/{k} tok1 sd',v[0][3][0],0,t2['tokens1']['sd'],'')
    c.chk(f'{p}/{k} tr0',v[1][0][0],0,100*t2['trunc0']['mean'],'trunc0'); c.chk(f'{p}/{k} tr0 sd',v[1][1][0],0,100*t2['trunc0']['sd'],'')
    c.chk(f'{p}/{k} tr1',v[1][2][0],0,100*t2['trunc1']['mean'],'trunc1'); c.chk(f'{p}/{k} tr1 sd',v[1][3][0],0,100*t2['trunc1']['sd'],'')
    c.chk(f'{p}/{k} dacc',v[2][0][0],1,t2['d_acc']['mean'],'d_acc'); c.chk(f'{p}/{k} dacc sd',v[2][1][0],1,t2['d_acc']['sd'],'')
    c.chk(f'{p}/{k} daccC',v[3][0][0],1,t2['d_acc_completed']['mean'],'d_acc_completed'); c.chk(f'{p}/{k} daccC sd',v[3][1][0],1,t2['d_acc_completed']['sd'],'')
c.report()
# ---- tab03
ph=J('phi_resample_s012')
c=Checker('tab03')
for p,s,r in cellrows('tab03.tex'):
    v=[tok(x) for x in r[2:]]
    if s.startswith('Teacher'):
        for i,l in enumerate(LAM): c.chk(f'{p} think {l}',v[i][0][0],0,100*tr[P(p)]['rates'][l],'think_rate')
        continue
    k=sname(s); rr=ph[f'{P(p)}/{FULL[k]}']['rows']
    for i,l in enumerate(LAM):
        c.chk(f'{p}/{k} R {l}',v[i][0][0],v[i][0][1],rr[l]['retention'],'phi_resample_s012 retention')
        if len(v[i])>1: c.chk(f'{p}/{k} R sd {l}',v[i][1][0],v[i][1][1],rr[l]['retention_sd'],'retention_sd')
        # recompute sd from seeds
        sd1=st.stdev(rr[l]['retention_seeds']); 
        if abs(sd1-rr[l]['retention_sd'])>1e-6: print('   note retention_sd != stdev(seeds) ddof1',p,k,l,rr[l]['retention_sd'],sd1)
        if abs(st.mean(rr[l]['retention_seeds'])-rr[l]['retention'])>1e-6: print('   note retention != mean(seeds)',p,k,l,rr[l]['retention'],st.mean(rr[l]['retention_seeds']))
c.report()
# ---- tab04
a1=J('alignment_s012'); a2=J('alignment_markers_s012'); a3=J('alignment_length_s012')
c=Checker('tab04')
for blk,(a,nm) in enumerate([(a1,'think'),(a2,'markers')]):
    pass
cur=None
for p,s,r in cellrows('tab04.tex'):
    pass
rs=rows('tab04.tex')
sig=None
for r in rs:
    if r[0].strip(): sig=r[0].strip()
    a=a1 if sig.startswith('Think') else (a2 if sig.startswith('Markers') else a3)
    pr=r[1].strip(); v=[tok(x) for x in r[2:]]
    if pr=='Pooled':
        c.chk(f'{sig} pooled mean',v[3][0][0],v[3][0][1],a['_summary']['pooled']['mean_r'],'_summary.pooled.mean_r')
        c.chk(f'{sig} pooled p',v[4][0][0],v[4][0][1],a['_summary']['pooled']['p'],'_summary.pooled.p')
        continue
    for j,k in enumerate(STUD):
        c.chk(f'{sig} {pr}/{k} r',v[j][0][0],2,a[f'{P(pr)}/{FULL[k]}']['r'],'r')
    c.chk(f'{sig} {pr} mean',v[3][0][0],2,a['_summary'][P(pr)]['mean_r'],'mean_r')
    c.chk(f'{sig} {pr} p',v[4][0][0],v[4][0][1],a['_summary'][P(pr)]['p'],'p')
c.report()
print('alignment summary keys',list(a1['_summary'].keys()))
# ---- tab05
cp=J('consensus_pre'); cq=J('consensus_post')
c=Checker('tab05')
for p,s,r in cellrows('tab05.tex'):
    k=sname(s); key=f'{P(p)}/{FULL[k]}'; e=cp[key]; q=cq[key]; v=[tok(x) for x in r[2:]]
    c.chk(f'{p}/{k} switch',v[0][0][0],1,float(e['transition']),'transition')
    c.chk(f'{p}/{k} dims pre',v[1][0][0],1,e['dims'],'consensus_pre.dims')
    c.chk(f'{p}/{k} in_loo',v[2][0][0],3,e['in_loo'],'in_loo')
    sds=list(e['in_loo_by_seed'].values()); c.chk(f'{p}/{k} in_loo sd(ddof1)',v[2][1][0],3,st.stdev(sds),'stdev(in_loo_by_seed)')
    c.chk(f'{p}/{k} in_loo mean-of-seeds',v[2][0][0],3,st.mean(sds),'mean(in_loo_by_seed)')
    sds=list(e['out_by_seed'].values()); c.chk(f'{p}/{k} out',v[3][0][0],4,e['out'],'out'); c.chk(f'{p}/{k} out sd',v[3][1][0],4,st.stdev(sds),'stdev(out_by_seed)')
    c.chk(f'{p}/{k} chance',v[4][0][0],3,e['chance'],'chance')
    c.chk(f'{p}/{k} dims post',v[5][0][0],1,q['dims'],'consensus_post.dims')
    if q['transition']!=e['transition']: print('  note post transition differs',key)
c.report()
