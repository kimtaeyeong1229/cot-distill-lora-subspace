"""In-text ranges of the manuscript (abstract, body, Limitations, appendix text, captions) recomputed from the JSONs.
Each check: (location, text value as lo-hi, decimals, list of JSON values). A range passes when the rounded min/max
of the JSON values equal the quoted ends."""
from common import *
import math
ss=J('seed_spread'); me=J('main_edits'); bl=J('baselines'); ood=J('ood'); em=J('energy_match'); dg=J('degeneration')
lc=J('length_confound'); rv=J('review_stats'); sae=J('sae_llama'); px=J('phi_xseed'); ph=J('phi_resample_s012')
sup=J('supplementary_stats'); to=J('top_overlap'); al=J('alignment_s012'); aL=J('alignment_length_s012'); td=J('teacher_density')
pa=J('pairA_seed_ratio'); sr=J('switch_regression'); ra=J('region_anatomy'); cp=J('consensus_pre'); cq=J('consensus_post')
CELLS=[f'{p}/{k}' for p in PAIRS for k in STUD]
n=0; bad=[]
def R(loc,lo,hi,dec,vals,src):
    global n; n+=1
    tol=0.5*10**(-dec)+1e-9  # exact .5 ties may round either way
    ok=abs(min(vals)-lo)<=tol and abs(max(vals)-hi)<=tol
    if not ok: bad.append(f'{loc}: text {lo}-{hi} vs JSON {min(vals):.{dec+3}f}-{max(vals):.{dec+3}f} ({src})')
def V(loc,x,dec,val,src):
    global n; n+=1
    if abs(val-x)>0.5*10**(-dec)+1e-9: bad.append(f'{loc}: text {x} vs JSON {val:.{dec+3}f} ({src})')
pct=lambda a,b:100*a/b
# Intro / 5.1
R('intro/5.1 markers kept after removal (%)',0.2,10,1 if False else 0,[0.2],'') if False else None
R('5.1 kept % (0.2-10)',0.2,9.9,1,[pct(bl[c]['rm_pre']['markers']['mean'],bl[c]['base0']['markers']['mean']) for c in CELLS],'baselines rm_pre/base0 markers')
R('intro length shrink x',3.7,5.4,1,[bl[c]['base0']['tokens']['mean']/bl[c]['rm_pre']['tokens']['mean'] for c in CELLS],'baselines base0/rm_pre tokens')
R('5.1 cut markers',11,22,0,[bl[c]['budget']['markers']['mean'] for c in CELLS],'baselines.budget.markers')
R('5.1 removal markers',0.07,1.4,2 if False else 1,[bl[c]['rm_pre']['markers']['mean'] for c in CELLS],'baselines.rm_pre.markers') if False else None
R('5.1 top dims B,C',2.1,2.5,1,[to[c]['dims_top'] for c in CELLS if c[0] in 'BC'],'top_overlap.dims_top')
R('5.1 top inside B,C %',64,74,0,[100*to[c]['cover_top'] for c in CELLS if c[0] in 'BC'],'top_overlap.cover_top')
R('5.1 top kept B,C %',1,18,0,[pct(bl[c]['gs_energy']['markers']['mean'],bl[c]['base0']['markers']['mean']) for c in CELLS if c[0] in 'BC'],'baselines gs_energy/base0')
R('5.1 pair A top-removal markers',3.3,10.6,1,[bl[c]['gs_energy']['markers']['mean'] for c in CELLS if c[0]=='A'],'baselines.gs_energy.markers')
R('5.1 pair A region-removal markers',0.18,1.16,2,[bl[c]['rm_pre']['markers']['mean'] for c in CELLS if c[0]=='A'],'baselines.rm_pre.markers')
# 5.2
rg=rv['graft_recovery_early']; AC=[c for c in CELLS if c[0] in 'AC']
R('5.2/intro early recovery beta1 %',2,8,0,[100*rg[c]['beta1'] for c in AC],'review_stats.graft_recovery_early.beta1')
R('5.2/intro early recovery beta2 %',56,218,0,[100*rg[c]['beta2'] for c in AC],'review_stats.graft_recovery_early.beta2')
rec=[100*(ss[c]['edits']['graft1']['markers_per_1k']['mean']-ss[c]['table1']['1.0']['mean'])/(ss[c]['table1']['0.0']['mean']-ss[c]['table1']['1.0']['mean']) for c in CELLS if c!='B/OLMo']
R('5.2 whole recovery beta1 (8 cells) %',1,16,0,rec,'seed_spread graft1 vs table1')
R('5.2 other beta2 early markers',0.08,3.4,2 if False else 1,[ss[c]['edits']['graft_other2']['head_markers_per_1k']['mean'] for c in CELLS],'') if False else None
R('5.2 beta1 completion A,C %',48,94,0,[100*ss[c]['edits']['graft1']['completed']['mean'] for c in AC],'seed_spread.edits.graft1.completed')
R('5.2 beta2 completion region %',0,23,0,[100*ss[c]['edits']['graft2']['completed']['mean'] for c in CELLS],'graft2.completed')
R('5.2 beta2 completion whole %',0,30,0,[100*ss[c]['edits']['graft_whole2']['completed']['mean'] for c in CELLS],'graft_whole2.completed')
R('5.2 beta2 completion other %',55,91,0,[100*ss[c]['edits']['graft_other2']['completed']['mean'] for c in CELLS],'graft_other2.completed')
R('5.2 region graft loops %',69,96,0,[100*dg[c]['graft2']['mean'] for c in CELLS],'degeneration.graft2')
R('5.2 lambda0 truncation (reach the limit) %',51,87,0,[100*ss[c]['table2']['trunc0']['mean'] for c in CELLS],'seed_spread.table2.trunc0')
R('Limitations lambda0 loops %',14,80,0,[100*dg[c]['base0']['mean'] for c in CELLS],'degeneration.base0')
# 5.3
h=lambda c,e: ss[c]['edits'][e]['head_markers_per_1k']['mean']/ss[c]['base']['base0_head_markers_per_1k']['mean']
w=lambda c,e: ss[c]['edits'][e]['markers_per_1k']['mean']/ss[c]['table1']['0.0']['mean']
tk=lambda c,e: 100*(ss[c]['edits'][e]['avg_tokens']['mean']/ss[c]['table2']['tokens0']['mean']-1)
R('5.3 early ratio gamma1,4',0.96,1.58,2,[h(c,e) for c in CELLS for e in ('ampn1','ampn4')],'seed_spread head ampn1/ampn4')
R('5.3 early ratio all-in-region',0.87,1.79,2,[h(c,'ampV') for c in CELLS],'seed_spread head ampV')
R('5.3 whole kept % (excl C/Qwen)',26,100,0,[100*w(c,e) for c in CELLS if c!='C/Qwen' for e in ('ampn1','ampn4')],'seed_spread markers ampn1/4')
R('5.3 C/Qwen whole ratio',1.3,1.5,1,[w('C/Qwen',e) for e in ('ampn1','ampn4')],'')
R('5.3 tokens gamma4 %',-6,31,0,[tk(c,'ampn4') for c in CELLS],'ampn4 avg_tokens')
R('5.3 tokens all-in %',-11,34,0,[tk(c,'ampV') for c in CELLS],'ampV avg_tokens')
V('5.3 acc gamma4',10.1,1,100*rv['amplification']['nine_cell']['ampn4']['accuracy'],'review_stats.amplification.nine_cell')
R('App A unreachable % (per seed)',12,81,0,[100*em[c]['seeds'][s]['unreachable'] for c in CELLS for s in em[c]['seeds']],'')
R('App A ratio B (per seed)',0.67,1.07,2,[em[c]['seeds'][s]['ratio_B'] for c in CELLS for s in em[c]['seeds']],'')
R('App A ratio dW (per seed)',0.46,0.98,2,[em[c]['seeds'][s]['ratio_dW'] for c in CELLS for s in em[c]['seeds']],'')
R('App A dims region (per seed)',7.2,8.4,1,[em[c]['seeds'][s]['dims_region'] for c in CELLS for s in em[c]['seeds']],'')
# 5.4
R('5.4 removal SAE kept %',13,24,0,[100*sae[p]['lambda0.0_rm_pre']['sae']['mean']/sae[p]['base0.0']['sae']['mean'] for p in PAIRS],'sae_llama')
R('5.4 other SAE kept %',92,106,0,[100*sae[p]['lambda0.0_rm_decisive']['sae']['mean']/sae[p]['base0.0']['sae']['mean'] for p in PAIRS],'sae_llama')
sb=rv['sae_subsets']
R('5.4 cue graft2 level A,C %',97,101,0,[100*sb[p]['cue']['graft2_level'] for p in 'AC'],'review_stats.sae_subsets')
R('5.4 verify graft2 level A,C %',52,53,0,[100*sb[p]['verify']['graft2_level'] for p in 'AC'],'')
R('5.4 verify masked A,C %',50,50,0,[100*sb[p]['verify_masked']['graft2_level'] for p in 'AC'],'')
V('5.4 B verify lambda1 %',24,0,100*sb['B']['verify']['lambda1_level'],''); V('5.4 B verify graft2 %',51,0,100*sb['B']['verify']['graft2_level'],'')
for p,x in zip('AC',(0.86,0.91)): V(f'5.4 masked r {p}',x,2,sae[p]['corr_keyword_sae_masked'],'sae_llama.corr_keyword_sae_masked')
# 6.1
for b,(lo,hi),(olo,ohi) in (('gsm8k',(0.2,9.2),(50,197)),('gpqa_diamond',(0.1,10.4),(27,140))):
    R(f'6.1 {b} shared kept %',lo,hi,1,[pct(ood[b][c]['rm_pre']['markers']['mean'],ood[b][c]['base0']['markers']['mean']) for c in CELLS],'ood')
    R(f'6.1 {b} other kept %',olo,ohi,0,[pct(ood[b][c]['rm_other']['markers']['mean'],ood[b][c]['base0']['markers']['mean']) for c in CELLS],'ood')
R('6.1 GSM8K truncation lambda0 %',11,84,0,[100*ood['gsm8k'][c]['base0']['truncated']['mean'] for c in CELLS],'ood.gsm8k.base0.truncated')
# 6.2 / App E
R('6.2 two-seed overlap %',54,60,0,[100*J('threshold_sensitivity')[c]['two_seed_region']['0.75']['overlap'] for c in CELLS],'threshold_sensitivity')
te=sup['two_seed_energy']
R('App E two-seed lambda0 energy %',43,67,0,[100*te[c]['share_0.0'] for c in CELLS],'supplementary_stats.two_seed_energy')
R('App E two-seed lambda1 energy %',1,10,0,[100*te[c]['share_1.0'] for c in CELLS],'')
R('App E chance %',0.2,2.3,1,[100*te[c]['chance_share'] for c in CELLS],'')
# 4.x / App B
R('4.2 R corr outside A/Qwen',0.80,0.95,2,[al[f'{P(c[0])}/{FULL[c[2:]]}']['r'] for c in CELLS if c!='A/Qwen'],'alignment_s012.r')
R('4.2 length vs think-drop corr',0.64,0.78,2,[aL['_summary'][p]['r_length_vs_think_drop'] for p in ('A','B','D')],'alignment_length_s012._summary')
V('4.2 length pooled r',0.91,2,aL['_summary']['pooled']['mean_r'],''); V('4.2 length pooled p',0.0002,4,aL['_summary']['pooled']['p'],''); V('4.2 length A r',0.95,2,aL['_summary']['A']['mean_r'],'')
R('4.2 single-pair p (think, markers)',0.05,0.20,2,[a['_summary'][p]['p'] for a in (al,J('alignment_markers_s012')) for p in ('A','B','D')],'alignment_*_s012._summary.p')
print('  NOTE single-pair p with length:',{p:round(aL['_summary'][p]['p'],4) for p in ('A','B','D')})
iv=lambda p,i: lc['pairs'][p]['intervals'][i]
R('4.3 pair A held kept',0.50,0.58,2,list(iv('A',0)['kept_vs_half_reference'].values()),'length_confound')
R('4.3 pair A switch kept',0.12,0.19,2,list(iv('A',1)['kept_vs_half_reference'].values()),'')
pu=lambda p,i,k: -math.log(iv(p,i)['kept_vs_half_reference'][k])/iv(p,i)['log_len_drop']
R('4.3 A switch/held per unit',2.46,2.62,2,[pu('A',1,k)/pu('A',0,k) for k in STUD],'')
R('App B A held per unit',0.81,1.03,2,[pu('A',0,k) for k in STUD],''); R('App B A switch per unit',2.07,2.62,2,[pu('A',1,k) for k in STUD],'')
R('App B B,C held per unit',0.22,1.23,2,[pu(p,i,k) for p in 'BC' for i in range(5) if iv(p,i)['class']=='held' for k in STUD],'')
R('App B first switch steps',0.98,1.36,2,[pu(p,2,k) for p in 'BC' for k in STUD],''); R('App B C second step',1.87,2.99,2,[pu('C',3,k) for k in STUD],'')
R('4.3 B 0.6-0.8 kept',0.43,0.50,2,list(iv('B',3)['kept_vs_half_reference'].values()),''); R('4.3 B 0.6-0.8 per unit',1.35,1.67,2,[pu('B',3,k) for k in STUD],''); R('4.3 B switch per unit',1.18,1.35,2,[pu('B',2,k) for k in STUD],'')
held=[v for p in PAIRS for i in range(5) if iv(p,i)['class']=='held' for v in iv(p,i)['kept_vs_half_reference'].values()]
sw=[v for p in PAIRS for i in range(5) if iv(p,i)['class']=='switch' for v in iv(p,i)['kept_vs_half_reference'].values()]
R('4.3 held kept all',0.50,0.96,2,held,''); R('4.3 switch kept all',0.12,0.56,2,sw,'')
R('4.3 held kept B,C',0.78,0.96,2,[v for p in 'BC' for i in range(5) if iv(p,i)['class']=='held' for v in iv(p,i)['kept_vs_half_reference'].values()],'')
R('App B per-marker switch A',1.05,1.33,2,[-math.log(iv('A',1)['kept_vs_half_reference'][k])/iv('A',1)['marker_log_ratio'] for k in STUD],'')
R('App B per-marker held A',0.47,0.61,2,[-math.log(iv('A',0)['kept_vs_half_reference'][k])/iv('A',0)['marker_log_ratio'] for k in STUD],'')
R('App B seed ratio',2.43,2.70,2,[v for c in pa for v in pa[c]['switch_over_held']['all']],'')
V('App B regression switch',0.46,2,sr['switch'],'switch_regression'); R('App B regression lower ends',-1.53,-0.11,2,[v[0] for v in sr['switch_ci95_by_bootstrap_seed'].values()],'')
th=rv['thresholds']; V('App B held max',10.0,1,th['held_max_change'],'review_stats.thresholds'); R('App B first step drops',62.0,90.5,1,th['switch_first_step_drops'],''); V('App B 2nd step',26.8,1,th['switch_second_step_drops'][0],'')
S=lc['summary']
V('App B R per log held',-0.36,2,S['held']['R_change_per_log_len']['mean'],''); V('App B R per log switch',-0.52,2,S['switch']['R_change_per_log_len']['mean'],'')
L=lc['leave_lambda_out_region']
R('App B split 0.4',0.48,0.55,2,[L[c]['by_left_out_lambda']['0.4']['0.4'] for c in L if c[0] in 'BC'],''); R('App B split 0.6',0.19,0.30,2,[L[c]['by_left_out_lambda']['0.4']['0.6'] for c in L if c[0] in 'BC'],'')
R('App B same-seed 0.2',0.31,0.68,2,[v['0.2']['same'] for v in px.values()],'phi_xseed'); R('App B xseed 0.2',0.11,0.21,2,[v['0.2']['xseed'] for v in px.values()],'')
for p,a,b,(l1,h1),(l2,h2) in (('A','0.2','0.4',(0.11,0.13),(0.04,0.04)),('B','0.4','0.6',(0.15,0.17),(0.07,0.08)),('D','0.4','0.8',(0.16,0.18),(0.03,0.05))):
    R(f'App B xseed {p} {a}',l1,h1,2,[v[a]['xseed'] for k,v in px.items() if k[0]==p],'phi_xseed'); R(f'App B xseed {p} {b}',l2,h2,2,[v[b]['xseed'] for k,v in px.items() if k[0]==p],'')
# App A
tt=sup['train_think_rate']
for p,x in zip(PAIRS,(9,29,6)): V(f'App A OLMo truncated {p}',x,0,100*sup['olmo_over_4096'][p]['0.0']['share'],'supplementary_stats.olmo_over_4096')
for p,x in zip(PAIRS,(2113,2261,1015)): V(f'3.1 n_train {p}',x,0,lc['pairs'][p]['n_train'],'length_confound.pairs.n_train')
# Tables-in-text
R('4.3 pre dims',5.4,7.7,1,[cp[k]['dims'] for k in cp],'consensus_pre.dims'); R('4.3 coverage pre %',41,57,0,[100*cp[k]['in_loo'] for k in cp],'consensus_pre.in_loo')
R('4.3 coverage post %',5,19,0,[100*cp[k]['out'] for k in cp],''); R('4.3 chance %',0.5,1.8,1,[100*cp[k]['chance'] for k in cp],'')
R('intro energy share %',47,72,0,[100*ra[c]['overall']['share_0.0'] for c in CELLS],'region_anatomy'); R('4.3 lambda1 share %',1,15,0,[100*ra[c]['overall']['share_1.0'] for c in CELLS],'')
R('App A anatomy dims',6.4,8.0,1,[ra[c]['overall']['dims'] for c in CELLS],'')
R('App D lambda1 control length x',0.62,1.17,2,[ss[c]['edits']['rm_pre_l1']['avg_tokens']['mean']/ss[c]['table2']['tokens1']['mean'] for c in CELLS],'seed_spread rm_pre_l1/tokens1')
R('Table 14 caption OLMo other-removed completion %',48,65,0,[100*me[c]['lambda0.0_rm_decisive']['completed'] for c in CELLS if c.endswith('OLMo')],'main_edits'); R('Table 14 caption OLMo unedited %',39,49,0,[100*(1-me[c]['base0']['truncated']) for c in CELLS if c.endswith('OLMo')],'')
R('4.1 student r',0.82,1.00,2,[td[p]['corr'][k] for p in PAIRS for k in STUD],'teacher_density.corr'); V('4.1 pooled r',0.86,2,td['pooled_corr'],'')
R('4.1 shorter x',2,7,0,[ss[c]['table2']['tokens0']['mean']/ss[c]['table2']['tokens1']['mean'] for c in CELLS],'seed_spread.table2')
R('4.1 acc gain pp',1.7,19.2,1,[ss[c]['table2']['d_acc']['mean'] for c in CELLS],'seed_spread.table2.d_acc')
# ---- added with the appendix compaction (2026-10-05)
lr=J('lambda0_retention')
for p,l,lo,hi in (('A','1.0',0.067,0.086),('B','1.0',0.155,0.166),('C','1.0',0.065,0.136)):
    R(f'App B lambda0 region kept {p} at {l}',lo,hi,3,[lr[f'{p}/{k}']['kept'][l] for k in STUD],'lambda0_retention.<cell>.kept')
R('Table 11 caption markers kept %',0.15,9.93,2,[pct(bl[c]['rm_pre']['markers']['mean'],bl[c]['base0']['markers']['mean']) for c in CELLS],'baselines rm_pre/base0 markers')
R('App D lambda1 control markers before',0.07,5.14,2,[ss[c]['table1']['1.0']['mean'] for c in CELLS],'seed_spread.table1.1.0')
R('App D lambda1 control markers after',0.03,4.03,2,[ss[c]['edits']['rm_pre_l1']['markers_per_1k']['mean'] for c in CELLS],'seed_spread.edits.rm_pre_l1')
R('Table 4 caption top-removal tokens',733,3914,0,[bl[c]['gs_energy']['tokens']['mean'] for c in CELLS],'baselines.gs_energy.tokens')
R('Table 4 caption region-removal tokens',732,1659,0,[bl[c]['rm_pre']['tokens']['mean'] for c in CELLS],'baselines.rm_pre.tokens')
gb=J('graft_beta')
V('App D beta4/8 region done max %',1,0,max(100*gb[c][f'graft{b}']['completed'] for c in CELLS for b in (4,8)),'graft_beta graft4/8 completed')
V('App D beta4/8 other/whole done max %',49,0,max(100*gb[c][f'{e}{b}']['completed'] for c in CELLS for e in ('graft_orth','graft_full') for b in (4,8) if f'{e}{b}' in gb[c]),'graft_beta')
V('App D beta4/8 accuracy max %',8.0,1,max(100*gb[c][k]['accuracy'] for c in CELLS for k in gb[c] if k[-1] in '48' and isinstance(gb[c][k],dict) and 'accuracy' in gb[c][k]),'graft_beta')
sr2=rv['graft_seed_recovery']
# the early-marker ratio is read from the manuscript itself: every printed "0.87--x" must equal the JSON maximum
import re, glob
txt=' '.join(open(f).read() for f in glob.glob('/home/lab/workspace/NAACL_cot_distill_lora_subspace/sections/*.tex'))
hi=max(h(c,'ampV') for c in CELLS)
for m in re.finditer(r'0\.87--(\d\.\d\d)',txt): V('printed early-marker ratio upper end (0.87--x)',float(m.group(1)),2,hi,'seed_spread head ampV / base0 head')
sd_=rv['split_difficulty']; V('Limitations level-5 share before filter %',30,0,100*sd_['A']['level5_share_all'],'review_stats.split_difficulty.level5_share_all')
R('Limitations level-5 share after filter %',16,17,0,[100*sd_[p]['level5_share_kept'] for p in PAIRS],'review_stats.split_difficulty.level5_share_kept')
print(f'{n} in-text values/ranges checked, {len(bad)} mismatches')
for b in bad: print('  MISMATCH',b)
