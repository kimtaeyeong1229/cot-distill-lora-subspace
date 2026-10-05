from common import *
ss=J('seed_spread'); bl=J('baselines'); sae=J('sae_llama'); ood=J('ood'); bs=J('base_students')
def ms(c,label,toks,d,scale=1,src=''):
    c.chk(label+' mean',toks[0][0],toks[0][1],scale*d['mean'],src)
    c.chk(label+' sd',toks[1][0],toks[1][1],scale*d['sd'],src)
# tab11 (SAE features) is checked in t_tables.py; the lines below print the ratios quoted in Sec. 5.4
for p in PAIRS:
    b0=sae[p]['base0.0']['sae']['mean']; s=sae[p]
    print(f'  {p}: corr_keyword_sae {s["corr_keyword_sae"]:.3f}; masked/base0 {s["base0.0"]["sae_masked"]["mean"]/b0:.3f}; rm_pre/b0 {s["lambda0.0_rm_pre"]["sae"]["mean"]/b0:.3f}; rm_other/b0 {s["lambda0.0_rm_decisive"]["sae"]["mean"]/b0:.3f}; ampn1 {s["ampn1"]["sae"]["mean"]/b0:.3f} ampn4 {s["ampn4"]["sae"]["mean"]/b0:.3f} ampV {s["ampV"]["sae"]["mean"]/b0:.3f} (sd {s["ampV"]["sae"]["sd"]:.3f}); graft2 {s["graft2"]["sae"]["mean"]:.3f} vs base1 {s["base1.0"]["sae"]["mean"]:.3f}; rm_scale/b0 {s["lambda0.0_rm_scale"]["sae"]["mean"]/b0:.3f}')
    k0=s['base0.0']['keyword']['mean']
    print(f'     keyword ratios: rm_other {s["lambda0.0_rm_decisive"]["keyword"]["mean"]/k0:.3f} ampn1 {s["ampn1"]["keyword"]["mean"]/k0:.3f} ampn4 {s["ampn4"]["keyword"]["mean"]/k0:.3f} ampV {s["ampV"]["keyword"]["mean"]/k0:.3f}; graft_orth2 head? ')
# old Tables 13-15 (tab12-14) became Figures 5 and 6, which read ood.json and baselines.json directly; the audit checks the
# numbers their captions print and that the nine-cell means drawn in Figure 6(a) are those of Table 1 (t_new.py checks Table 1)
import re
APP=open(SECTIONS+'09_appendix.tex').read()
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
# Figure 5 (per-cell ways to turn thinking off) was removed from the manuscript on 2026-10-06; Table 3 keeps the nine-cell means
