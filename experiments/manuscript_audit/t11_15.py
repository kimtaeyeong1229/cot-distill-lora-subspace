from common import *
ss=J('seed_spread'); bl=J('baselines'); sae=J('sae_llama'); ood=J('ood'); bs=J('base_students')
def ms(c,label,toks,d,scale=1,src=''):
    c.chk(label+' mean',toks[0][0],toks[0][1],scale*d['mean'],src)
    c.chk(label+' sd',toks[1][0],toks[1][1],scale*d['sd'],src)
# tab11 (Table 13 right block, transposed: rows are conditions, columns pairs A, B, C; seed means, last row r with markers)
c=Checker('Table 10 bottom (tab11)')
rowkey={'$\\lambda=0$':('base0.0','sae'),'$\\lambda=1$':('base1.0','sae'),'Shared removed':('lambda0.0_rm_pre','sae'),
        'Other removed':('lambda0.0_rm_decisive','sae'),'Graft shared ($\\beta=2$)':('graft2','sae'),'Graft other ($\\beta=2$)':('graft_orth2','sae'),
        'All in region':('ampV','sae'),'$\\lambda=0$, markers deleted':('base0.0','sae_masked')}
for r in rows('tab11.tex',0):
    name=r[0].strip(); v=[tok(x) for x in r[1:]]
    if name.startswith('$r$'):
        for i,p in enumerate(PAIRS): c.chk(f'{p} r with markers',v[i][0][0],2,sae[p]['corr_keyword_sae'],'sae_llama.corr_keyword_sae')
        continue
    k,f=rowkey[name]
    for i,p in enumerate(PAIRS): c.chk(f'{p} {k}.{f}',v[i][0][0],v[i][0][1],sae[p][k][f]['mean'],f'sae_llama.{k}.{f}.mean')
c.report()
for p in PAIRS:
    b0=sae[p]['base0.0']['sae']['mean']; s=sae[p]
    print(f'  {p}: corr_keyword_sae {s["corr_keyword_sae"]:.3f}; masked/base0 {s["base0.0"]["sae_masked"]["mean"]/b0:.3f}; rm_pre/b0 {s["lambda0.0_rm_pre"]["sae"]["mean"]/b0:.3f}; rm_other/b0 {s["lambda0.0_rm_decisive"]["sae"]["mean"]/b0:.3f}; ampn1 {s["ampn1"]["sae"]["mean"]/b0:.3f} ampn4 {s["ampn4"]["sae"]["mean"]/b0:.3f} ampV {s["ampV"]["sae"]["mean"]/b0:.3f} (sd {s["ampV"]["sae"]["sd"]:.3f}); graft2 {s["graft2"]["sae"]["mean"]:.3f} vs base1 {s["base1.0"]["sae"]["mean"]:.3f}; rm_scale/b0 {s["lambda0.0_rm_scale"]["sae"]["mean"]/b0:.3f}')
    k0=s['base0.0']['keyword']['mean']
    print(f'     keyword ratios: rm_other {s["lambda0.0_rm_decisive"]["keyword"]["mean"]/k0:.3f} ampn1 {s["ampn1"]["keyword"]["mean"]/k0:.3f} ampn4 {s["ampn4"]["keyword"]["mean"]/k0:.3f} ampV {s["ampV"]["keyword"]["mean"]/k0:.3f}; graft_orth2 head? ')
# old Tables 13-15 (tab12-14) became Figures 5 and 6, which read ood.json and baselines.json directly; the audit checks the
# numbers their captions print and that the nine-cell means drawn in Figure 6(a) are those of Table 1 (t_new.py checks Table 1)
import re
APP=open('/home/lab/workspace/NAACL_cot_distill_lora_subspace/sections/09_appendix.tex').read()
def caption(label):
    i=APP.index('\\label{'+label+'}'); j=APP.rindex('\\caption{',0,i); return APP[j:i]
CELLS=[f'{p}/{k}' for p in PAIRS for k in STUD]
c=Checker('Figure 4 caption (fig:ood)')
cap=caption('fig:ood')
m=re.search(r'leaves ([\d.]+)--([\d.]+)\\% of the markers on GSM8K and ([\d.]+)--([\d.]+)\\% on GPQA-Diamond and cuts GSM8K truncation from ([\d.]+)--([\d.]+)\\% to ([\d.]+)--([\d.]+)\\%',cap)
g=[float(x) for x in m.groups()]
kept=lambda b:[100*ood[b][x]['rm_pre']['markers']['mean']/ood[b][x]['base0']['markers']['mean'] for x in CELLS]
c.chk('GSM8K kept min',g[0],1,min(kept('gsm8k')),'ood.gsm8k rm_pre/base0'); c.chk('GSM8K kept max',g[1],0,max(kept('gsm8k')),'')
c.chk('GPQA kept min',g[2],1,min(kept('gpqa_diamond')),'ood.gpqa_diamond rm_pre/base0'); c.chk('GPQA kept max',g[3],0,max(kept('gpqa_diamond')),'')
tr=lambda cond:[100*ood['gsm8k'][x][cond]['truncated']['mean'] for x in CELLS]
c.chk('GSM8K trunc lambda0 min',g[4],0,min(tr('base0')),'ood.gsm8k.base0.truncated'); c.chk('GSM8K trunc lambda0 max',g[5],0,max(tr('base0')),'')
c.chk('GSM8K trunc removed min',g[6],1,min(tr('rm_pre')),'ood.gsm8k.rm_pre.truncated'); c.chk('GSM8K trunc removed max',g[7],1,max(tr('rm_pre')),'')
c.report()
c=Checker('Figure 5 caption (fig:off)')
cap=caption('fig:off')
U={'Qwen':bs['Qwen2.5-0.5B-Instruct'],'Llama':bs['Llama-3.2-1B-Instruct'],'OLMo':bs['OLMo-2-1B-Instruct']}
above=[x for x in CELLS if bl[x]['rmseed']['accuracy']['mean']>U[x.split('/')[1]]['accuracy']['mean']]
c.chk('two-seed removal above untrained (cells)',2,0,len(above),'baselines.rmseed vs base_students')
if set(above)!={'A/OLMo','B/Llama'}: c.err.append(f'cells above untrained {above} (caption names A/OLMo and B/Llama)')
m=re.search(r'gives ([\d.]+)--([\d.]+)\\%',cap); mg=[100*bl[x]['merge0.5']['accuracy']['mean'] for x in CELLS]
c.chk('merged min',float(m.group(1)),1,min(mg),'baselines.merge0.5.accuracy'); c.chk('merged max',float(m.group(2)),1,max(mg),'')
c.report()
