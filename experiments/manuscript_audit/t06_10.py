"""Transposed one-column appendix tables (rows conditions, columns the nine cells A/Q ... C/O; seed means):
Table 9 removal (tab06), Table 12 grafts (tab08), Table 14 amplification (tab10), Table 13 loops and beta 2/4/8 (tab17),
Table 7 removal vs untrained (tab26). Rows that only repeated another table (lambda=1 and lambda=0 tokens, the beta=2 lambda=0 early markers) were cut. Seed SDs moved to the JSON; the seed spread of the main edits is Table 9 (tab19, t_new.py)."""
from common import *
ss=J('seed_spread'); bl=J('baselines'); me=J('main_edits'); rv=J('review_stats'); em=J('energy_match'); dg=J('degeneration'); gb=J('graft_beta')
ru=J('removal_vs_untrained')
E=lambda c,e,f: ss[c]['edits'][e][f]['mean']
def check(c,fname,spec):
    """spec: {(section-prefix, label): (fn(cell) -> value, scale, source, mean_fn or None)}"""
    seen=set()
    for sec,lab,cells in trows(fname):
        key=None
        for (sp,lb),v in spec.items():
            if sec.startswith(sp) and lab==lb: key=(sp,lb); break
        if key is None: c.err.append(f'{fname}: no audit rule for row [{sec}] {lab}'); continue
        seen.add(key); fn,sc,src,mfn=spec[key]
        vals=[tok(x) for x in cells[:9]]
        for cell,v in zip(QLO,vals): c.chk(f'{lab} {cell}',v[0][0],v[0][1],sc*fn(cell),src)
        if mfn and len(cells)>9 and cells[9].strip(): t=tok(cells[9])[0]; c.chk(f'{lab} mean',t[0],t[1],mfn(),src+' (nine-cell mean)')
    for k in spec:
        if k not in seen: c.err.append(f'{fname}: row {k} missing')
    c.report()
nine=lambda f: st.mean(f(x) for x in QLO)
# ---- Table 8 removal (tab06)
c=Checker('Table 9 (tab06)')
W='Whole'; Ea='Early'; A='MATH'
spec={(W,'$\\lambda=0$'):(lambda x: ss[x]['table1']['0.0']['mean'],1,'seed_spread.table1.0.0',None),
      (W,'Removed'):(lambda x: E(x,'rm_pre','markers_per_1k'),1,'seed_spread.edits.rm_pre.markers_per_1k',None),
      (W,'Cut'):(lambda x: bl[x]['budget']['markers']['mean'],1,'baselines.budget.markers',None),
      (W,'Other'):(lambda x: E(x,'rm_other','markers_per_1k'),1,'seed_spread.edits.rm_other',None),
      (W,'Top'):(lambda x: bl[x]['gs_energy']['markers']['mean'],1,'baselines.gs_energy.markers',None),
      (W,'Scaled'):(lambda x: E(x,'rm_scale','markers_per_1k'),1,'seed_spread.edits.rm_scale',None)}
for lab,cn in (('$\\lambda=0$','base0'),('Removed','lambda0.0_rm_pre'),('Cut','lambda0.0_budget'),('Other','lambda0.0_rm_decisive'),('Top','lambda0.0_gs_energy'),('Scaled','lambda0.0_rm_scale')):
    spec[(Ea,lab)]=((lambda cn: lambda x: me[x][cn]['head_markers_per_1k'])(cn),1,f'main_edits.{cn}.head_markers_per_1k',None)
accf={'$\\lambda=0$':lambda x: ss[x]['base']['base0_accuracy']['mean'],'Removed':lambda x: E(x,'rm_pre','accuracy'),'Cut':lambda x: bl[x]['budget']['accuracy']['mean'],
      'Other':lambda x: E(x,'rm_other','accuracy'),'Top':lambda x: bl[x]['gs_energy']['accuracy']['mean'],'Scaled':lambda x: E(x,'rm_scale','accuracy')}
for lab,f in accf.items(): spec[(A,lab)]=(f,100,'seed_spread / baselines accuracy',(lambda f: lambda: 100*nine(f))(f))
check(c,'tab06.tex',spec)
# ---- Table 12 grafts (tab08)
c=Checker('Table 12 (tab08)')
B1='$\\beta=1$'; B2='$\\beta=2$'; D='Completion'
def rec(x):
    g=E(x,'graft1','markers_per_1k'); l1=ss[x]['table1']['1.0']['mean']; l0=ss[x]['table1']['0.0']['mean']; return 100*(g-l1)/(l0-l1)
spec={(B1,'$\\lambda=1$'):(lambda x: ss[x]['table1']['1.0']['mean'],1,'seed_spread.table1.1.0',None),
      (B1,'Shared'):(lambda x: E(x,'graft1','markers_per_1k'),1,'edits.graft1',None),(B1,'Other'):(lambda x: E(x,'graft_other1','markers_per_1k'),1,'edits.graft_other1',None),
      (B1,'Whole'):(lambda x: E(x,'graft_whole1','markers_per_1k'),1,'edits.graft_whole1',None),
      (B1,'Tok.\\ shared'):(lambda x: E(x,'graft1','avg_tokens'),1,'edits.graft1.avg_tokens',None),
      (B1,'Rec.\\ (\\%)'):(rec,1,'recovered whole-response gap from seed means',None),
      (B1,'Rec.\\ low'):(lambda x: rv['graft1_whole_recovery'][x]['min'],100,'review_stats.graft1_whole_recovery.min',None),
      (B1,'Rec.\\ high'):(lambda x: rv['graft1_whole_recovery'][x]['max'],100,'review_stats.graft1_whole_recovery.max',None),
      (B2,'$\\lambda=1$'):(lambda x: ss[x]['base']['base1_head_markers_per_1k']['mean'],1,'base1_head',None),
      (B2,'Shared'):(lambda x: E(x,'graft2','head_markers_per_1k'),1,'graft2 head',None),(B2,'Other'):(lambda x: E(x,'graft_other2','head_markers_per_1k'),1,'graft_other2 head',None),
      (B2,'Whole'):(lambda x: E(x,'graft_whole2','head_markers_per_1k'),1,'graft_whole2 head',None),
      (D,'Done sh.\\ $\\beta=1$'):(lambda x: E(x,'graft1','completed'),100,'edits.graft1.completed',None),
      (D,'Done shared'):(lambda x: E(x,'graft2','completed'),100,'edits.graft2.completed',None),(D,'Done other'):(lambda x: E(x,'graft_other2','completed'),100,'edits.graft_other2.completed',None),
      (D,'Done whole'):(lambda x: E(x,'graft_whole2','completed'),100,'edits.graft_whole2.completed',None),
      (D,'Acc.\\ shared'):(lambda x: E(x,'graft2','accuracy'),100,'edits.graft2.accuracy',None),(D,'Acc.\\ other'):(lambda x: E(x,'graft_other2','accuracy'),100,'edits.graft_other2.accuracy',None),
      (D,'Acc.\\ whole'):(lambda x: E(x,'graft_whole2','accuracy'),100,'edits.graft_whole2.accuracy',None)}
check(c,'tab08.tex',spec)
# ---- Table 14 amplification (tab10)
c=Checker('Table 14 (tab10)')
Wm='Whole'; Em='Early'; Am='Accuracy'
spec={(Wm,'$\\lambda=0$'):(lambda x: ss[x]['table1']['0.0']['mean'],1,'table1.0.0',None),
      (Wm,'Tok.\\ all'):(lambda x: E(x,'ampV','avg_tokens'),1,'edits.ampV.avg_tokens',None),
      (Em,'$\\lambda=0$'):(lambda x: ss[x]['base']['base0_head_markers_per_1k']['mean'],1,'base0_head',None),
      (Em,'Gain $\\gamma=4$'):(lambda x: em[x]['mean']['gain_ampn4'],1,'energy_match.mean.gain_ampn4',None)}
for lab,e in (('$\\gamma=1$','ampn1'),('$\\gamma=4$','ampn4'),('All','ampV')):
    spec[(Wm,lab)]=((lambda e: lambda x: E(x,e,'markers_per_1k'))(e),1,f'edits.{e}.markers_per_1k',None)
    spec[(Em,lab)]=((lambda e: lambda x: E(x,e,'head_markers_per_1k'))(e),1,f'edits.{e}.head',None)
for lab,cn in (('$\\lambda=0$','base0'),('$\\gamma=1$','ampn1'),('$\\gamma=4$','ampn4'),('All','ampV')):
    for pre,f in (('Acc.\\ ','accuracy'),('Done ','completed')):
        spec[(Am,pre+lab)]=((lambda cn,f: lambda x: rv['amplification'][x][cn][f])(cn,f),100,f'review_stats.amplification.{cn}.{f}',(lambda cn,f: lambda: 100*rv['amplification']['nine_cell'][cn][f])(cn,f))
check(c,'tab10.tex',spec)
# ---- Table 13 loops and beta 2/4/8 (tab17)
c=Checker('Table 13 (tab17)')
Lp='Responses'; Mk='Whole'; Bt='Region'
spec={}
for lab,e in (('$\\lambda=0$','base0'),('$\\lambda=1$','base1'),('Removed','rm_pre'),('G1 shared','graft1'),('G2 shared','graft2'),('G2 other','graft_other2'),('G2 whole','graft_whole2'),('$\\gamma=4$','ampn4'),
              ('Other rem.','rm_other'),('Scaled','rm_scale'),('Top rem.','rm_top'),('Cut','cut')):
    spec[(Lp,lab)]=((lambda e: lambda x: dg[x][e]['mean'])(e),100,f'degeneration.{e}',None)
spec[(Mk,'G2 shared')]=(lambda x: E(x,'graft2','markers_per_1k'),1,'edits.graft2.markers_per_1k',None)
spec[(Mk,'G2 other')]=(lambda x: E(x,'graft_other2','markers_per_1k'),1,'edits.graft_other2.markers_per_1k',None)
dk={2:'graft2',4:'graft4_seed0',8:'graft8_seed0'}
for b in (2,4,8):
    spec[(Bt,f'$\\beta={b}$ Markers')]=((lambda b: lambda x: gb[x][f'graft{b}']['head_markers_per_1k'])(b),1,f'graft_beta.graft{b}.head_markers_per_1k',None)
    spec[(Bt,f'$\\beta={b}$ Done')]=((lambda b: lambda x: gb[x][f'graft{b}']['completed'])(b),100,f'graft_beta.graft{b}.completed',None)
    spec[(Bt,f'$\\beta={b}$ Acc.')]=((lambda b: lambda x: gb[x][f'graft{b}']['accuracy'])(b),100,f'graft_beta.graft{b}.accuracy',None)
    def lp(x,b=b):
        v=dg[x]['graft2']['seeds'][0] if b==2 else dg[x][dk[b]]
        return v['mean'] if isinstance(v,dict) else v
    spec[(Bt,f'$\\beta={b}$ Loop')]=(lp,100,f'degeneration.{dk[b]} (seed 0)',None)
check(c,'tab17.tex',spec)
# ---- Table 11 removal vs untrained (tab26)
c=Checker('Table 7 (tab26)')
kap=lambda d: d['vs_untrained']['response_level']['cohen_kappa']
un=lambda x: ru['untrained'][x.split('/')[1]]
Ac='Accuracy'; Ka='$\\kappa$'
spec={(Ac,'Acc.\\ untr.'):(lambda x: un(x)['accuracy'],100,'untrained.accuracy',None),(Ac,'Acc.\\ rem.'):(lambda x: ru['cells'][x]['rm_pre']['accuracy'],100,'rm_pre.accuracy',None),
      (Ac,'Acc.\\ $\\lambda=1$'):(lambda x: ru['cells'][x]['lambda1']['accuracy'],100,'lambda1.accuracy',None),(Ac,'Tok.\\ untr.'):(lambda x: un(x)['tokens'],1,'untrained.tokens',None),
      (Ac,'Tok.\\ rem.'):(lambda x: ru['cells'][x]['rm_pre']['tokens'],1,'rm_pre.tokens',None),(Ac,'Okay (\\%)'):(lambda x: ru['cells'][x]['rm_pre']['opens_okay'],100,'rm_pre.opens_okay',None),
      (Ka,'$\\kappa$ rem.'):(lambda x: kap(ru['cells'][x]['rm_pre']),1,'rm_pre kappa',None),(Ka,'$\\kappa$ $\\lambda=0$'):(lambda x: kap(ru['cells'][x]['lambda0']),1,'lambda0 kappa',None),
      (Ka,'$\\kappa$ $\\lambda=1$'):(lambda x: kap(ru['cells'][x]['lambda1']),1,'lambda1 kappa',None),(Ka,'$\\kappa$ self'):(lambda x: un(x)['self_ceiling']['cohen_kappa'],1,'self_ceiling kappa',None)}
check(c,'tab26.tex',spec)
