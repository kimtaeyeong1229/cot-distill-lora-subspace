"""Checks for the blocks of the compacted appendix (printed numbers after the 2026-10-05 one-column layout):
Table 10 early markers (tab06 block 1), Table 11 bottom (tab09 block 1, beta=1 and Whole), Table 12 (tab10) middle/bottom,
Table 1 (tab15), Table 15 control loops (tab17 block 1) and its beta 2/4/8 block (tab18), Table 16 (tab19) with the
per-seed graft recovery columns (old Table 28), Table 4 (tab23, with the old Table 19 top-direction columns), Table 5 (tab24),
Table 9 (tab25), Table 13 (tab26), Table 7 (tab27) and Table 14 top (tab29; tab11 is in t11_15.py)."""
from common import *
ss=J('seed_spread'); me=J('main_edits'); sy=J('symmetric_R'); rv=J('review_stats'); em=J('energy_match')
dg=J('degeneration'); gb=J('graft_beta'); es=J('edit_stats'); pb=J('paired_bootstrap'); ra=J('region_anatomy')
bl=J('baselines'); ood=J('ood'); bs=J('base_students'); ex=J('examples')
LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
def ms(c,label,toks,d,scale=1,src=''):
    c.chk(label+' mean',toks[0][0],toks[0][1],scale*d['mean'],src)
    c.chk(label+' sd',toks[1][0],toks[1][1],scale*d['sd'],src)
# --- Table 1 (tab15)
c=Checker('Table 1 (tab15)')
def m9(f): return st.mean(f(p,k) for p in PAIRS for k in STUD)
for row in rows('tab15.tex',0):
    key={'$\\lambda=0$':'base0','Shared removed':'rm_pre','Other removed':'rm_other','$\\lambda=1$':'base1'}[row[0].strip()]
    v=[tok(x)[0] for x in row[1:]]
    for bi,b in enumerate(('gsm8k','gpqa_diamond')):
        c.chk(f'top {key} {b} tokens',v[2*bi][0],0,m9(lambda p,k: ood[b][f'{p}/{k}'][key]['tokens']['mean']),f'ood.{b}.{key}.tokens nine-cell')
        c.chk(f'top {key} {b} acc',v[2*bi+1][0],1,m9(lambda p,k: 100*ood[b][f'{p}/{k}'][key]['accuracy']['mean']),f'ood.{b}.{key}.accuracy nine-cell')
SFULL={'Qwen':'Qwen2.5-0.5B-Instruct','Llama':'Llama-3.2-1B-Instruct','OLMo':'OLMo-2-1B-Instruct'}
spec={'$\\lambda=0$':'base0','Shared removed':'rm_pre','Two-seed region removed':'rmseed','Gold-Switch-style removal':'gs_tau','Length-matched cut':'budget',
      'Think-skip (adapted)':'nothink','Other removed (two-seed region)':'rmseed_other','Merged with $\\lambda=1$':'merge0.5','$\\lambda=1$':'base1','Untrained':None}
for row in rows('tab15.tex',1):
    name=row[0].strip(); v=[tok(x)[0] for x in row[1:]]; key=spec[name]
    if key:
        mk=m9(lambda p,k: bl[f'{p}/{k}'][key]['markers']['mean']); tk=m9(lambda p,k: bl[f'{p}/{k}'][key]['tokens']['mean']); ac=m9(lambda p,k: 100*bl[f'{p}/{k}'][key]['accuracy']['mean'])
    else:
        mk=m9(lambda p,k: bs[SFULL[k]]['markers_first_sample']['mean']); tk=m9(lambda p,k: bs[SFULL[k]]['avg_tokens']['mean']); ac=m9(lambda p,k: 100*bs[SFULL[k]]['accuracy']['mean'])
    c.chk(f'bottom {name} markers',v[0][0],1,mk,f'baselines.{key}.markers'); c.chk(f'bottom {name} tokens',v[1][0],0,tk,f'baselines.{key}.tokens'); c.chk(f'bottom {name} acc',v[2][0],1,ac,f'baselines.{key}.accuracy')
c.report()
# --- Table 9 (tab19): block 0 removal, block 1 graft and gamma=4, block 2 recovered early-marker gap; Cell labels, summary rows by label
c=Checker('Table 10 (tab19)')
f0=['rm_retained','rm_acc','rm_minus_top','rm_minus_cut']; f1=['graft2_head','graft2_done','amp4_retained']
pbk={'rm_minus_top':'top','rm_minus_cut':'cut'}
f1=f1+['rec1','rec2']   # block 1 now also carries the recovered early-marker gap at beta = 1 and 2 (cell value, seed min, seed max)
for blk,fs in ((0,f0),(1,f1)):
    for r in rows('tab19.tex',blk):
        lab=r[0].strip(); v=[tok(x) for x in r[1:]]
        if lab not in es['cells']:
            for j,f in enumerate(fs):
                if not v[j]: continue
                n=es['nine_cell'][f]; d=v[j][0][1]
                if 'Nine-cell' in lab: c.chk(f'nine mean {f}',v[j][0][0],d,n['mean'],'edit_stats.nine_cell.mean')
                elif 'cells fixed' in lab:
                    ci=pb['nine_cell'][pbk[f]]['ci95_pp'] if f in pbk else n['ci95']
                    c.chk(f'ci lo {f}',v[j][0][0],d,ci[0],'paired_bootstrap / edit_stats ci95'); c.chk(f'ci hi {f}',v[j][1][0],d,ci[1],'')
                elif 'cells as units' in lab:
                    ci=rv['cell_interval'][pbk[f]]['ci95']; c.chk(f'cell ci lo {f}',v[j][0][0],d,ci[0],'review_stats.cell_interval.ci95'); c.chk(f'cell ci hi {f}',v[j][1][0],d,ci[1],'')
                elif 'Median' in lab: c.chk(f'median {f}',v[j][0][0],d,n['median'],'median')
                elif 'w/o' in lab: c.chk(f'mean w/o {f}',v[j][0][0],d,n['mean_without_B_C_Qwen'],'mean_without_B_C_Qwen')
                else: c.err.append(f'unknown summary row {lab}')
            continue
        e=es['cells'][lab]
        for j,f in enumerate(fs):
            d=v[j][0][1]
            if f in ('rec1','rec2'):
                bk='beta1' if f=='rec1' else 'beta2'; cv=rv['graft_recovery_early'][lab][bk]; sr_=rv['graft_seed_recovery'][lab][bk]
                c.chk(f'{lab} {bk} recovery',v[j][0][0],d,100*cv,'review_stats.graft_recovery_early')
                c.chk(f'{lab} {bk} seed min',v[j][1][0],d,100*sr_['min'],'graft_seed_recovery.min'); c.chk(f'{lab} {bk} seed max',v[j][2][0],d,100*sr_['max'],'graft_seed_recovery.max')
                continue
            c.chk(f'{lab} {f}',v[j][0][0],d,e[f]['value'],f'edit_stats.cells.{f}.value'); c.chk(f'{lab} {f} min',v[j][1][0],d,e[f]['min'],'min'); c.chk(f'{lab} {f} max',v[j][2][0],d,e[f]['max'],'max')
c.report()
print('  paired_bootstrap nine_cell', {k:(round(v['diff_pp'],2) if 'diff_pp' in v else None, [round(x,2) for x in v['ci95_pp']]) for k,v in pb['nine_cell'].items()})
print('  review cell_interval', {k:(round(v['mean'],2),[round(x,2) for x in v['ci95']]) for k,v in rv['cell_interval'].items()})
pos={t:sum(1 for c_ in pb['cells'].values() if c_[t]['ci95_pp'][0]>0) for t in ('cut','top','other')}
poss={t:sum(1 for c_ in pb['cells'].values() if c_[t]['ci95_pp_seeds_resampled'][0]>0) for t in ('cut','top','other')}
print('  per-cell CI above zero (cut,top,other):',pos,' seeds resampled:',poss)
print('  cells with top CI>0:',[k for k,c_ in pb['cells'].items() if c_['top']['ci95_pp'][0]>0])
# --- Table 2 (tab23: anatomy + top directions, 12 columns; tab24: energy audit + gains, 10 columns)
to=J('top_overlap')
c=Checker('Table 2 (tab23, tab24)')
for r in rows('tab23.tex',0):
    cell=r[0].strip(); a=ra[cell]; v=[tok(x)[0] for x in r[1:]]; t=to[cell]
    c.chk(f'{cell} dims',v[0][0],1,a['overall']['dims'],'region_anatomy.overall.dims')
    c.chk(f'{cell} share0',v[1][0],0,100*a['overall']['share_0.0'],'overall.share_0.0'); c.chk(f'{cell} share1',v[2][0],0,100*a['overall']['share_1.0'],'overall.share_1.0')
    c.chk(f'{cell} chance',v[3][0],1,100*a['overall']['chance_share'],'overall.chance_share')
    for j,dpt in enumerate(('early','middle','late')): c.chk(f'{cell} {dpt}',v[4+j][0],0,100*a['by_depth'][dpt]['share_0.0'],f'by_depth.{dpt}.share_0.0')
    c.chk(f'{cell} top dims',v[7][0],1,t['dims_top'],'top_overlap.dims_top'); c.chk(f'{cell} region spanned',v[8][0],3,t['cover_region'],'top_overlap.cover_region')
    c.chk(f'{cell} top inside',v[9][0],3,t['cover_top'],'top_overlap.cover_top'); c.chk(f'{cell} top chance',v[10][0],4,t['chance'],'top_overlap.chance')
for r in rows('tab24.tex',0):
    cell=r[0].strip(); m=em[cell]['mean']; v=[tok(x)[0] for x in r[1:]]
    for j,(f,sc,d) in enumerate([('region_B',1,2),('ratio_B',1,2),('ratio_dW',1,2),('unreachable',100,0),('dims_region',1,1),('dims_other',1,1),('gain_ampn1',1,2),('gain_ampn4',1,2),('gain_ampV',1,2)]):
        c.chk(f'{cell} {f}',v[j][0],d,sc*m[f],f'energy_match.{cell}.mean.{f}')
c.report()
# --- Table 25 (tab25)
c=Checker('Table 15 (tab25)')
rs=rows('tab25.tex'); allr=ex['students']+ex['teachers']
for r,e in zip(rs,allr):
    r[0]=r[0].replace('\\addlinespace','').strip()
    if r[0].strip()!=e['row']: c.err.append(f'row name {r[0]} vs {e["row"]}')
    c.chk(f'{e["row"]} tokens',tok(r[2])[0][0],0,e['tokens'],'examples.tokens'); c.chk(f'{e["row"]} markers',tok(r[3])[0][0],1,e['markers_per_1k'],'examples.markers_per_1k')
    if 'status' in e and r[4].strip()!=e['status']: c.err.append(f'{e["row"]} status {r[4]} vs {e["status"]}')
c.report()

# --- Table 5 bottom block (tab27): teacher structure at the ends of the held intervals
ts=J('teacher_structure')
c=Checker('Table 8 (tab27)')
colmap=[('A','0.0'),('A','0.2'),('A','0.4'),('B','0.0'),('B','0.4'),('C','0.0'),('C','0.4')]
fld={'Markers':(lambda d:d['a_markers_per_1k_all_generations'],1),'Words':(lambda d:d['a_mean_words'],0),'Tag (\\%)':(lambda d:100*d['b_think_tag'],0),'Recap (\\%)':(lambda d:100*d['b_tagged_recap'],0),
     'Opener (\\%)':(lambda d:100*d['e_opens_monologue'],0),'Sent. (\\%)':(lambda d:100*d['e_monologue_sentence_share'],0),
     'Verif.':(lambda d:d['c_markers_per_1k']['verification'],2),'Check (\\%)':(lambda d:100*d['d_any_verification'],0),
     'Hesit.':(lambda d:d['c_markers_per_1k']['hesitation'],2),'Altern.':(lambda d:d['c_markers_per_1k']['alternative'],2),
     'Attempts':(lambda d:d['c_attempts_per_solution'],2)}
for r in rows('tab27.tex'):
    name=r[0].strip().replace('\\%','\\%')
    key=[f_ for f_ in fld if f_.replace('\\\\','\\')==name or f_==name]
    if not key: print('  (not audited) tab27 row',name); continue
    fn,d=fld[key[0]]
    for j,(p,l) in enumerate(colmap): c.chk(f'{p} {l} {name}',tok(r[1+j])[0][0],d,fn(ts[p]['per_lambda'][l]),'teacher_structure.per_lambda')
c.report()
# --- Table 13 left block (tab29): nine-cell student structure on the first 300 words (recap on completed responses)
sst=J('student_structure')['summary']['all']['level']
c=Checker('Table 11 top (tab29)')
conds=['lambda0','rm_pre','rm_decisive','graft2','lambda1','untrained']
rowf={'Hesitation':lambda h,w:h['c_markers_per_1k']['hesitation'],'Verification':lambda h,w:h['c_markers_per_1k']['verification'],
      'Attempts':lambda h,w:h['c_attempts_per_solution'],'Opener':lambda h,w:h['e_opens_monologue'],
      'Mono.\\ sent.':lambda h,w:h['e_monologue_sentence_share'],'Recap (compl.)':lambda h,w:w.get('b_boxed_recap_completed')}
for r in rows('tab29.tex'):
    name=r[0].strip(); fn=rowf[name]
    for j,cn in enumerate(conds):
        cell=r[1+j].strip()
        if cell in ('--','-'): continue
        val=fn(sst[cn]['head'],sst[cn]['whole']); c.chk(f'{name} {cn}',tok(cell)[0][0],tok(cell)[0][1],val,'student_structure.summary.all.level')
c.report()
