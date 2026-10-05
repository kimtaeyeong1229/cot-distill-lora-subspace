from common import *
import numpy as np
tr=J('think_rate'); td=J('teacher_density'); ss=J('seed_spread')
LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
# ---- numbers of the removed Figure 4 (lambda sweep): marker r per cell is the r column of Table 3 (tab02, checked below);
# pooled r in the Table 3 caption; teacher markers 15.5 -> 5.0 in Table 7 (tab27, t_new.py) and its caption; B/Llama seeds in the
# Table 8 caption; R_sym and the A/Qwen B(lambda) collapse in App B text
import re
c=Checker('Former Figure 4 numbers (Table 5, 10 and 11 captions, App B text)')
cap=caption('tab:2'); m=re.search(r'\(([\d.]+) over all 54 points\)',cap); c.chk('pooled r (Table 5 caption)',float(m.group(1)),2,td['pooled_corr'],'teacher_density.pooled_corr')
cap=caption('tab:6'); m=re.search(r'seeds of ([\d.]+), ([\d.]+) and (\d+\.\d)',cap)
for i,x in enumerate(m.groups()): c.chk(f'B/Llama lambda0 seed {i} (Table 11 caption)',float(x),1,ss['B/Llama']['table1']['0.0']['seeds'][i],'seed_spread.table1 seeds')
cap=caption('tab:structure'); m=re.search(r'fall from ([\d.]+) to ([\d.]+)',cap)
c.chk('A teacher markers 0.0 (Table 10 caption)',float(m.group(1)),1,td['A']['teacher']['0.0'],'teacher_density'); c.chk('A teacher markers 0.2',float(m.group(2)),1,td['A']['teacher']['0.2'],'teacher_density')
APP=open(APP_TEX).read()
m=re.search(r'pair A\'s first interval (?:then )?drops (?:by )?([\d.]+)--([\d.]+) against ([\d.]+)--([\d.]+) at the switch',APP)
rs=J('symmetric_R'); keyR=lambda p,k: f'{P(p)}/{FULL[k]}'
d1=[rs[keyR('A',k)]['drops']['0.0->0.2'] for k in STUD]; d2=[rs[keyR('A',k)]['drops']['0.2->0.4'] for k in STUD]
g=[float(x) for x in m.groups()]
c.chk('R_sym A first drop min',g[0],2,min(d1),'symmetric_R drops'); c.chk('R_sym A first drop max',g[1],2,max(d1),'')
c.chk('R_sym A switch drop min',g[2],2,min(d2),''); c.chk('R_sym A switch drop max',g[3],2,max(d2),'')
c.chk('R_sym largest drop at switch (cells, App B "all nine")',9,0,sum(1 for k in rs if '/' in k and rs[k].get('largest_at_switch')),'symmetric_R.largest_at_switch')
ph=J('phi_resample_s012'); rq=ph['A/Qwen2.5-0.5B']['rows']
m=re.search(r'(?:collapses|falls) from ([\d.]+) at \$\\lambda=0.2\$ to ([\d.]+) at 0.4 and ([\d.]+) at 1',APP)
for x,l in zip(m.groups(),('0.2','0.4','1.0')): c.chk(f'A/Qwen B({l}) (App B)',float(x),2,rq[l]['baseline'],'phi_resample_s012 rows.baseline')
m=re.search(r'stays (?:at )?([\d.]+)--([\d.]+) after the switch',APP); rv_=[rq[l]['retention'] for l in ('0.4','0.6','0.8','1.0')]
c.chk('A/Qwen R after switch min',float(m.group(1)),2,min(rv_),'retention'); c.chk('A/Qwen R after switch max',float(m.group(2)),2,max(rv_),'')
c.report()
# ---- tab02 (Table 3, one block): Cell | tokens l0 -> l1 | truncated l0 -> l1 | dAcc all | dAcc completed | accuracy at six lambda | marker r
md=J('main_density')
c=Checker('Table 5 (tab02)')
for r in rows('tab02.tex',0):
    cell=r[0]; t2=ss[cell]['table2']; v=[tok(x) for x in r[1:]]; p,k=cell.split('/')
    c.chk(f'{cell} tok0',v[0][0][0],0,t2['tokens0']['mean'],'table2.tokens0'); c.chk(f'{cell} tok1',v[0][1][0],0,t2['tokens1']['mean'],'table2.tokens1')
    c.chk(f'{cell} tr0',v[1][0][0],0,100*t2['trunc0']['mean'],'trunc0'); c.chk(f'{cell} tr1',v[1][1][0],0,100*t2['trunc1']['mean'],'trunc1')
    c.chk(f'{cell} dacc',v[2][0][0],1,t2['d_acc']['mean'],'d_acc'); c.chk(f'{cell} daccC',v[3][0][0],1,t2['d_acc_completed']['mean'],'d_acc_completed')
    for i,l in enumerate(LAM): c.chk(f'{cell} acc {l}',v[4+i][0][0],1,100*md[cell][l]['accuracy'],f'main_density.{cell}.{l}.accuracy')
    c.chk(f'{cell} marker r',v[10][0][0],2,td[p]['corr'][k],'teacher_density.corr')
c.report()
# ---- tab04
a1=J('alignment_s012'); a2=J('alignment_markers_s012'); a3=J('alignment_length_s012'); tc=J('think_rate_completed')
c=Checker('Table 6 (tab04)')
for blk,(a,nm) in enumerate([(a1,'think'),(a2,'markers')]):
    pass
cur=None
for p,s,r in cellrows('tab04.tex'):
    pass
rs=rows('tab04.tex')
sig=None
for r in rs:
    if r[0].strip(): sig=r[0].strip()
    pr=r[1].strip(); v=[tok(x) for x in r[2:]]
    if sig.startswith(('Completed','Training')):   # think rate on completed solutions / training problems
        a=tc['alignment']['completed' if sig.startswith('Completed') else 'train']
        cells_,summ=a['cells'],a['summary']
    else:
        a=a1 if sig.startswith('Think') else (a2 if sig.startswith('Markers') else a3)
        cells_,summ=a,a['_summary']
    if pr=='Pooled':
        c.chk(f'{sig} pooled mean',v[3][0][0],v[3][0][1],summ['pooled']['mean_r'],'pooled.mean_r')
        c.chk(f'{sig} pooled p',v[4][0][0],v[4][0][1],summ['pooled']['p'],'pooled.p')
        continue
    for j,k in enumerate(STUD):
        c.chk(f'{sig} {pr}/{k} r',v[j][0][0],2,cells_[f'{P(pr)}/{FULL[k]}']['r'],'r')
    c.chk(f'{sig} {pr} mean',v[3][0][0],2,summ[P(pr)]['mean_r'],'mean_r')
    c.chk(f'{sig} {pr} p',v[4][0][0],v[4][0][1],summ[P(pr)]['p'],'p')
c.report()
print('alignment summary keys',list(a1['_summary'].keys()))
# ---- tab05 (Table 5 top, one block): Cell | switch | dims pre | dims post | coverage LOO | post | chance | two-seed overlap 0.65/0.75/0.85
cp=J('consensus_pre'); cq=J('consensus_post'); th=J('threshold_sensitivity')
c=Checker('Table 7 top (tab05)')
key=lambda cell: f"{P(cell[0])}/{FULL[cell.split('/')[1]]}"
for r in rows('tab05.tex',0):
    cell=r[0]; e=cp[key(cell)]; q=cq[key(cell)]; v=[tok(x) for x in r[1:]]
    c.chk(f'{cell} switch',v[0][0][0],1,float(e['transition']),'transition')
    c.chk(f'{cell} dims pre',v[1][0][0],1,e['dims'],'consensus_pre.dims')
    c.chk(f'{cell} dims post',v[2][0][0],1,q['dims'],'consensus_post.dims')
    c.chk(f'{cell} in_loo',v[3][0][0],3,e['in_loo'],'in_loo'); c.chk(f'{cell} in_loo mean-of-seeds',v[3][0][0],3,st.mean(e['in_loo_by_seed'].values()),'mean(in_loo_by_seed)')
    c.chk(f'{cell} out',v[4][0][0],4,e['out'],'out')
    c.chk(f'{cell} chance',v[5][0][0],3,e['chance'],'chance')
    for i,cut in enumerate(('0.65','0.75','0.85')): c.chk(f'{cell} two-seed {cut}',v[6+i][0][0],0,100*th[cell]['two_seed_region'][cut]['overlap'],'threshold_sensitivity.two_seed_region.overlap')
c.report()
