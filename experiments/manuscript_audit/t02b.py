# Table 6 accuracy-at-every-lambda columns are checked in t01_05.py (tab02 is one block since the appendix compaction);
# this script keeps the derived checks: peak lambda per cell and the untrained comparison quoted in Limitations
from common import *
md=J('main_density'); LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
peak={}
for p in PAIRS:
    for k in STUD:
        acc={l:100*md[f'{p}/{k}'][l]['accuracy'] for l in LAM}; m=max(acc.values()); peak[f'{p}/{k}']=[l for l in LAM if abs(acc[l]-m)<1e-9]
c=Checker('App C accuracy peak (six of nine at 0.8, B/OLMo tied with 1, never before the switch)')
c.chk('cells peaking at 0.8',6,0,sum(1 for v in peak.values() if '0.8' in v),'main_density')
c.chk('cells peaking at 0.8 or 1',9,0,sum(1 for v in peak.values() if set(v)<= {'0.8','1.0'}),'main_density')
c.report()
print('  peak lambda per cell (exact):',peak)
bs=J('base_students'); U={'Qwen':100*bs['Qwen2.5-0.5B-Instruct']['accuracy']['mean'],'Llama':100*bs['Llama-3.2-1B-Instruct']['accuracy']['mean'],'OLMo':100*bs['OLMo-2-1B-Instruct']['accuracy']['mean']}
below=[(p,k,l) for p in PAIRS for k in STUD for l in LAM if 100*md[f'{p}/{k}'][l]['accuracy']<U[k]]
c=Checker('Limitations: below untrained at 41 of 54 points'); c.chk('points below untrained',41,0,len(below),'main_density vs base_students'); c.report()
