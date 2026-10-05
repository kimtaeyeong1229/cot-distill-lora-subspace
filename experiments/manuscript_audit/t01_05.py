from common import *
import numpy as np
tr=J('think_rate'); td=J('teacher_density'); ss=J('seed_spread')
LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
# ---- numbers of the removed Figure 4 (lambda sweep): marker r per cell is the r column of Table 3 (tab02, checked below);
# pooled r in the Table 3 caption; teacher markers 15.5 -> 5.0 in Table 14 (tab27, t_new.py) and its caption; B/Llama seeds in the
# Table 12 caption; R_sym and the A/Qwen B(lambda) collapse in App B text. The tables themselves are checked in t_tables.py.
import re
c=Checker('Former Figure 4 numbers (Tables 10 and 11 captions, App B text)')
cap=caption('tab:2'); m=re.search(r'\(([\d.]+) over all 54 points\)',cap); c.chk('pooled r (Table 8 caption)',float(m.group(1)),2,td['pooled_corr'],'teacher_density.pooled_corr')
cap=caption('tab:6'); m=re.search(r'seeds of ([\d.]+), ([\d.]+) and (\d+\.\d)',cap)
for i,x in enumerate(m.groups()): c.chk(f'B/Llama lambda0 seed {i} (Table 6 caption)',float(x),1,ss['B/Llama']['table1']['0.0']['seeds'][i],'seed_spread.table1 seeds')
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
