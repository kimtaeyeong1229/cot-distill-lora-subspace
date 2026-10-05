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
# --- Table 8 (tab06) early block
c=Checker('Table 10 early markers (tab06 block 1)')
cond=['base0','lambda0.0_rm_pre','lambda0.0_budget','lambda0.0_rm_decisive','lambda0.0_gs_energy','lambda0.0_rm_scale']
for p,s,r in cellrows('tab06.tex',1):
    k=sname(s)
    for i,cn in enumerate(cond): c.chk(f'{p}/{k} {cn}',tok(r[2+i])[0][0],2,me[f'{p}/{k}'][cn]['head_markers_per_1k'],f'main_edits.{cn}.head_markers_per_1k')
c.report()
# --- Table 10 (tab09) bottom block
c=Checker('Table 11 bottom (tab09 block 1)')
E=lambda p,k,e,f: ss[f'{p}/{k}']['edits'][e][f]
for p,s,r in cellrows('tab09.tex',1):
    k=sname(s); v=[tok(x) for x in r[2:]]
    ms(c,f'{p}/{k} graft1 done',v[0],E(p,k,'graft1','completed'),100,'seed_spread.edits.graft1.completed')
    for j,e in enumerate(['graft2','graft_other2','graft_whole2']):
        ms(c,f'{p}/{k} {e} done',v[1+j],E(p,k,e,'completed'),100,f'seed_spread.edits.{e}.completed')
        ms(c,f'{p}/{k} {e} acc',v[4+j],E(p,k,e,'accuracy'),100,f'seed_spread.edits.{e}.accuracy')
    # cross-check with main_edits
    for e,m in (('graft1','graft1'),('graft_whole2','graft_full2'),('graft2','graft2'),('graft_other2','graft_orth2')):
        if abs(E(p,k,e,'completed')['mean']-me[f'{p}/{k}'][m]['completed'])>1e-6: print('  note completed differs',p,k,e)
c.report()
# --- Table 11 (tab10) middle and bottom
c=Checker('Table 12 middle/bottom (tab10 blocks 1-2)')
for p,s,r in cellrows('tab10.tex',1):
    k=sname(s); v=[tok(x) for x in r[2:]]; cs=ss[f'{p}/{k}']
    ms(c,f'{p}/{k} head l0',v[0],cs['base']['base0_head_markers_per_1k'],1,'base0_head')
    for j,e in enumerate(['ampn1','ampn4','ampV']): ms(c,f'{p}/{k} head {e}',v[1+j],E(p,k,e,'head_markers_per_1k'),1,f'edits.{e}.head')
    c.chk(f'{p}/{k} gain4',v[4][0][0],2,em[f'{p}/{k}']['mean']['gain_ampn4'],'energy_match.mean.gain_ampn4')
nine={x:[] for x in range(8)}
for p,s,r in cellrows('tab10.tex',2):
    v=[tok(x) for x in r[2:]] if not p.startswith('\\multi') else None
    if p.startswith('\\multicolumn'):
        mr=[tok(x)[0] for x in r[1:]]
        for j,(cn,f) in enumerate([(a,'accuracy') for a in ('base0','ampn1','ampn4','ampV')]+[(a,'completed') for a in ('base0','ampn1','ampn4','ampV')]):
            c.chk(f'nine {cn} {f}',mr[j][0],mr[j][1],100*rv['amplification']['nine_cell'][cn][f],'review_stats.amplification.nine_cell')
        continue
    k=sname(s); a=rv['amplification'][f'{p}/{k}']
    for j,(cn,f) in enumerate([(x,'accuracy') for x in ('base0','ampn1','ampn4','ampV')]+[(x,'completed') for x in ('base0','ampn1','ampn4','ampV')]):
        c.chk(f'{p}/{k} {cn} {f}',v[j][0][0],v[j][0][1],100*a[cn][f],'review_stats.amplification')
        nine[j].append(100*a[cn][f])
    # cross-check vs main_edits
    if abs(a['base0']['accuracy']-me[f'{p}/{k}']['base0']['accuracy'])>1e-9: print('  note base0 acc differs',p,k)
    if abs(a['base0']['completed']-(1-me[f'{p}/{k}']['base0']['truncated']))>1e-9: print('  note base0 completed differs',p,k)
c.report()
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
# --- Table 20 (tab17) bottom
c=Checker('Table 15 control loops (tab17 block 1)')
for p,s,r in cellrows('tab17.tex',1):
    k=sname(s); d=dg[f'{p}/{k}']
    for j,e in enumerate(['rm_pre','rm_other','rm_scale','rm_top','cut']): c.chk(f'{p}/{k} loop {e}',tok(r[2+j])[0][0],0,100*d[e]['mean'],f'degeneration.{e}.mean')
c.report()
# --- Table 21 (tab18)
c=Checker('Table 15 beta 2/4/8 block (tab18)')
dk={2:'graft2',4:'graft4_seed0',8:'graft8_seed0'}
for p,s,r in cellrows('tab18.tex',0):
    k=sname(s); g=gb[f'{p}/{k}']; v=[tok(x) for x in r[2:]]
    for i,b in enumerate((2,4,8)):
        gg=g[f'graft{b}']
        c.chk(f'{p}/{k} b{b} markers',v[4*i][0][0],1,gg['head_markers_per_1k'],f'graft_beta.graft{b}.head_markers_per_1k')
        c.chk(f'{p}/{k} b{b} done',v[4*i+1][0][0],0,100*gg['completed'],f'graft_beta.graft{b}.completed')
        c.chk(f'{p}/{k} b{b} acc',v[4*i+2][0][0],1,100*gg['accuracy'],f'graft_beta.graft{b}.accuracy')
        lp=dg[f'{p}/{k}']['graft2']['seeds'][0] if b==2 else dg[f'{p}/{k}'][dk[b]]
        lp=lp['mean'] if isinstance(lp,dict) else lp
        c.chk(f'{p}/{k} b{b} loop',v[4*i+3][0][0],0,100*lp,f'degeneration.{dk[b]}')
# the other/whole beta 4/8 block became a sentence in App. D (checked in text_checks.py)
c.report()
# --- Table 22 (tab19) both blocks
c=Checker('Table 16 (tab19)')
f0=['rm_retained','rm_acc','rm_minus_top','rm_minus_cut']; f1=['graft2_head','graft2_done','amp4_retained']
pbk={'rm_minus_top':'top','rm_minus_cut':'cut'}
for blk,fs in ((0,f0),(1,f1)):
    for p,s,r in cellrows('tab19.tex',blk):
        if p.startswith('\\multicolumn'):
            lab=p; v=[tok(x) for x in r[1:]]
            for j,f in enumerate(fs):
                if not v[j]: continue
                n=es['nine_cell'][f]; d=v[j][0][1]
                if 'Nine-cell' in lab: c.chk(f'nine mean {f}',v[j][0][0],d,n['mean'],'edit_stats.nine_cell.mean')
                elif 'cells fixed' in lab or 'bootstrap' in lab:
                    ci=pb['nine_cell'][pbk[f]]['ci95_pp'] if f in pbk else n['ci95']
                    c.chk(f'ci lo {f}',v[j][0][0],d,ci[0],'paired_bootstrap.nine_cell.ci95_pp' if f in pbk else 'edit_stats.nine_cell.ci95'); c.chk(f'ci hi {f}',v[j][1][0],d,ci[1],'')
                elif 'cells as units' in lab:
                    ci=rv['cell_interval'][pbk[f]]['ci95']; c.chk(f'cell ci lo {f}',v[j][0][0],d,ci[0],'review_stats.cell_interval.ci95'); c.chk(f'cell ci hi {f}',v[j][1][0],d,ci[1],'')
                elif 'Median' in lab: c.chk(f'median {f}',v[j][0][0],d,n['median'],'median')
                elif 'without' in lab: c.chk(f'mean w/o {f}',v[j][0][0],d,n['mean_without_B_C_Qwen'],'mean_without_B_C_Qwen')
            continue
        k=sname(s); v=[tok(x) for x in r[2:]]; e=es['cells'][f'{p}/{k}']
        for j,f in enumerate(fs):
            d=v[j][0][1]
            c.chk(f'{p}/{k} {f}',v[j][0][0],d,e[f]['value'],f'edit_stats.cells.{f}.value'); c.chk(f'{p}/{k} {f} min',v[j][1][0],d,e[f]['min'],'min'); c.chk(f'{p}/{k} {f} max',v[j][2][0],d,e[f]['max'],'max')
        if blk==1:   # recovered early-marker gap at beta = 1 and 2 (old Table 28): cell value (seed min, seed max)
            for j,bk in ((3,'beta1'),(4,'beta2')):
                d=v[j][0][1]; cv=rv['graft_recovery_early'][f'{p}/{k}'][bk]; sr_=rv['graft_seed_recovery'][f'{p}/{k}'][bk]
                c.chk(f'{p}/{k} {bk} recovery',v[j][0][0],d,100*cv,'review_stats.graft_recovery_early')
                c.chk(f'{p}/{k} {bk} seed min',v[j][1][0],d,100*sr_['min'],'review_stats.graft_seed_recovery.min')
                c.chk(f'{p}/{k} {bk} seed max',v[j][2][0],d,100*sr_['max'],'review_stats.graft_seed_recovery.max')
c.report()
print('  paired_bootstrap nine_cell', {k:(round(v['diff_pp'],2) if 'diff_pp' in v else None, [round(x,2) for x in v['ci95_pp']]) for k,v in pb['nine_cell'].items()})
print('  review cell_interval', {k:(round(v['mean'],2),[round(x,2) for x in v['ci95']]) for k,v in rv['cell_interval'].items()})
pos={t:sum(1 for c_ in pb['cells'].values() if c_[t]['ci95_pp'][0]>0) for t in ('cut','top','other')}
poss={t:sum(1 for c_ in pb['cells'].values() if c_[t]['ci95_pp_seeds_resampled'][0]>0) for t in ('cut','top','other')}
print('  per-cell CI above zero (cut,top,other):',pos,' seeds resampled:',poss)
print('  cells with top CI>0:',[k for k,c_ in pb['cells'].items() if c_['top']['ci95_pp'][0]>0])
# --- Table 4 top block (tab23, with the top-direction columns of the old Table 19)
to=J('top_overlap')
c=Checker('Table 4 (tab23, two blocks)')
for r in rows('tab23.tex',0):
    cell=r[0].strip(); a=ra[cell]; v=[tok(x)[0] for x in r[1:]]
    c.chk(f'{cell} dims',v[0][0],1,a['overall']['dims'],'region_anatomy.overall.dims')
    c.chk(f'{cell} share0',v[1][0],0,100*a['overall']['share_0.0'],'overall.share_0.0')
    c.chk(f'{cell} share1',v[2][0],0,100*a['overall']['share_1.0'],'overall.share_1.0')
    c.chk(f'{cell} chance',v[3][0],1,100*a['overall']['chance_share'],'overall.chance_share')
    for j,dpt in enumerate(('early','middle','late')): c.chk(f'{cell} {dpt}',v[4+j][0],0,100*a['by_depth'][dpt]['share_0.0'],f'by_depth.{dpt}.share_0.0')
    if max(('early','middle','late'),key=lambda d: a['by_depth'][d]['share_0.0'])!='late': print('  note: largest depth share not late in',cell)
for r in rows('tab23.tex',1):
    cell=r[0].strip(); t=to[cell]; v=[tok(x)[0] for x in r[1:]]
    c.chk(f'{cell} top dims',v[0][0],1,t['dims_top'],'top_overlap.dims_top'); c.chk(f'{cell} region spanned',v[1][0],3,t['cover_region'],'top_overlap.cover_region')
    c.chk(f'{cell} top inside',v[2][0],3,t['cover_top'],'top_overlap.cover_top'); c.chk(f'{cell} top chance',v[3][0],4,t['chance'],'top_overlap.chance')
c.report()
md=[a['by_module'][m]['dims'] for a in ra.values() for m in a['by_module']]
print('  module-type dims range',round(min(md),2),round(max(md),2))
# --- Table 24 (tab24)
c=Checker('Table 5 (tab24, two blocks)')
for blk,fs in ((0,[('region_B',1,2),('ratio_B',1,2),('ratio_dW',1,2),('unreachable',100,0),('dims_region',1,1),('dims_other',1,1)]),(1,[('gain_ampn1',1,2),('gain_ampn4',1,2),('gain_ampV',1,2)])):
    for r in rows('tab24.tex',blk):
        cell=r[0].strip(); m=em[cell]['mean']; v=[tok(x)[0] for x in r[1:]]
        for j,(f,sc,d) in enumerate(fs): c.chk(f'{cell} {f}',v[j][0],d,sc*m[f],f'energy_match.{cell}.mean.{f}')
c.report()
for f in ['ratio_B','ratio_dW','unreachable','dims_region','dims_other','gain_ampn1','gain_ampn4','gain_ampV']:
    mn=min(em[c_]['min'][f] for c_ in em if '/' in c_); mx=max(em[c_]['max'][f] for c_ in em if '/' in c_)
    mmn=min(em[c_]['mean'][f] for c_ in em if '/' in c_); mmx=max(em[c_]['mean'][f] for c_ in em if '/' in c_)
    print(f'  {f}: per-seed range {mn:.4f}-{mx:.4f}; cell-mean range {mmn:.4f}-{mmx:.4f}')
# --- Table 25 (tab25)
c=Checker('Table 9 (tab25)')
rs=rows('tab25.tex'); allr=ex['students']+ex['teachers']
for r,e in zip(rs,allr):
    r[0]=r[0].replace('\\addlinespace','').strip()
    if r[0].strip()!=e['row']: c.err.append(f'row name {r[0]} vs {e["row"]}')
    c.chk(f'{e["row"]} tokens',tok(r[2])[0][0],0,e['tokens'],'examples.tokens'); c.chk(f'{e["row"]} markers',tok(r[3])[0][0],1,e['markers_per_1k'],'examples.markers_per_1k')
    if 'status' in e and r[4].strip()!=e['status']: c.err.append(f'{e["row"]} status {r[4]} vs {e["status"]}')
c.report()

# --- Table 11 (tab26): removal vs untrained
ru=J('removal_vs_untrained')
c=Checker('Table 13 (tab26, two blocks)')
kap=lambda d: d['vs_untrained']['response_level']['cohen_kappa']
for r in rows('tab26.tex',0):
    cell=r[0].strip(); k=cell.split('/')[1]; e=ru['cells'][cell]; u=ru['untrained'][k]; v=[tok(x)[0] for x in r[1:]]
    for j,(val,d,src) in enumerate([(100*u['accuracy'],1,'untrained.accuracy'),(100*e['rm_pre']['accuracy'],1,'rm_pre.accuracy'),(100*e['lambda1']['accuracy'],1,'lambda1.accuracy'),
            (u['tokens'],0,'untrained.tokens'),(e['rm_pre']['tokens'],0,'rm_pre.tokens'),(100*e['rm_pre']['opens_okay'],0,'rm_pre.opens_okay')]):
        c.chk(f'{cell} {src}',v[j][0],d,val,'removal_vs_untrained '+src)
for r in rows('tab26.tex',1):
    cell=r[0].strip(); k=cell.split('/')[1]; e=ru['cells'][cell]; u=ru['untrained'][k]; v=[tok(x)[0] for x in r[1:]]
    for j,(val,src) in enumerate([(kap(e['rm_pre']),'rm_pre kappa'),(kap(e['lambda0']),'lambda0 kappa'),(kap(e['lambda1']),'lambda1 kappa'),(u['self_ceiling']['cohen_kappa'],'self_ceiling kappa')]):
        c.chk(f'{cell} {src}',v[j][0],2,val,'removal_vs_untrained '+src)
c.report()
# --- Table 5 bottom block (tab27): teacher structure at the ends of the held intervals
ts=J('teacher_structure')
c=Checker('Table 7 (tab27)')
colmap=[('A','0.0'),('A','0.2'),('A','0.4'),('B','0.0'),('B','0.4'),('C','0.0'),('C','0.4')]
fld={'Words':(lambda d:d['a_mean_words'],0),'Tag (\\%)':(lambda d:100*d['b_think_tag'],0),'Recap (\\%)':(lambda d:100*d['b_tagged_recap'],0),
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
c=Checker('Table 14 top (tab29)')
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
