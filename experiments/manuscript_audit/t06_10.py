from common import *
ss=J('seed_spread'); bl=J('baselines')
def E(p,k,ed,f): return ss[f'{p}/{k}']['edits'][ed][f]
def mm(c,label,tk,src):  # tk (mean,dec),(sd,dec) ; src dict mean/sd ; scale
    pass
def ms(c,label,toks,d,scale=1,src=''):
    c.chk(label+' mean',toks[0][0],toks[0][1],scale*d['mean'],src)
    c.chk(label+' sd',toks[1][0],toks[1][1],scale*d['sd'],src)
# tab06
c=Checker('Table 8 top (tab06 block 0)')
for p,s,r in cellrows('tab06.tex',0):
    k=sname(s); v=[tok(x) for x in r[2:]]; cs=ss[f'{p}/{k}']; b=bl[f'{p}/{k}']
    ms(c,f'{p}/{k} l0',v[0],cs['table1']['0.0'],1,'table1 0.0')
    ms(c,f'{p}/{k} rm_pre',v[1],E(p,k,'rm_pre','markers_per_1k'),1,'edits.rm_pre')
    ms(c,f'{p}/{k} budget',v[2],b['budget']['markers'],1,'baselines.budget')
    ms(c,f'{p}/{k} rm_other',v[3],E(p,k,'rm_other','markers_per_1k'),1,'edits.rm_other')
    ms(c,f'{p}/{k} gs_energy',v[4],b['gs_energy']['markers'],1,'baselines.gs_energy')
    ms(c,f'{p}/{k} rm_scale',v[5],E(p,k,'rm_scale','markers_per_1k'),1,'edits.rm_scale')
    # cross-check seed_spread vs baselines
    for e in ('rm_pre','rm_other','rm_scale'):
        if abs(E(p,k,e,'markers_per_1k')['mean']-b[e]['markers']['mean'])>1e-3: print('  note ss vs baselines differ',p,k,e)
# the old middle block (tokens, lambda=1 control) became App. D text (checked in text_checks.py); block 1 is now the
# early-marker block, checked in t_new.py
c.report()
# tab07
c=Checker('Table 8 bottom (tab07)')
cols=[('base0',None),('rm_pre','ss'),('budget','bl'),('rm_other','ss'),('gs_energy','bl'),('rm_scale','ss')]
acc={x[0]:[] for x in cols}
meanrow=None
for p,s,r in cellrows('tab07.tex'):
    v=[tok(x) for x in r[2:]]
    if p.startswith('\\multicolumn'): meanrow=[tok(x)[0] for x in r[1:]]; continue
    k=sname(s); cs=ss[f'{p}/{k}']; b=bl[f'{p}/{k}']
    for i,(nm,srcf) in enumerate(cols):
        d=cs['base']['base0_accuracy'] if nm=='base0' else (E(p,k,nm,'accuracy') if srcf=='ss' else b[nm]['accuracy'])
        ms(c,f'{p}/{k} {nm}',v[i],d,100,nm); acc[nm].append(100*d['mean'])
for i,(nm,_) in enumerate(cols): c.chk(f'Mean {nm}',meanrow[i][0],meanrow[i][1],st.mean(acc[nm]),'nine-cell mean')
# caption claims
best=max(acc,key=lambda n: st.mean(acc[n])); print('  highest mean:',best, {n:round(st.mean(a),2) for n,a in acc.items()})
print('  rm_pre > budget per cell:',[round(a-b,2) for a,b in zip(acc['rm_pre'],acc['budget'])])
print('  rm_pre - gs_energy per cell:',[round(a-b,2) for a,b in zip(acc['rm_pre'],acc['gs_energy'])])
c.report()
# tab08
c=Checker('Table 9 top (tab08)')
for p,s,r in cellrows('tab08.tex'):
    k=sname(s); v=[tok(x) for x in r[2:]]; cs=ss[f'{p}/{k}']
    ms(c,f'{p}/{k} l1',v[0],cs['table1']['1.0'],1,'table1 1.0')
    ms(c,f'{p}/{k} graft1',v[1],E(p,k,'graft1','markers_per_1k'),1,'graft1')
    ms(c,f'{p}/{k} graft_other1',v[2],E(p,k,'graft_other1','markers_per_1k'),1,'graft_other1')
    ms(c,f'{p}/{k} graft_whole1',v[3],E(p,k,'graft_whole1','markers_per_1k'),1,'graft_whole1')
    ms(c,f'{p}/{k} tok1',v[4][:2],cs['table2']['tokens1'],1,'tokens1')
    ms(c,f'{p}/{k} graft1 tok',v[4][2:],E(p,k,'graft1','avg_tokens'),1,'graft1 tokens')
    g=E(p,k,'graft1','markers_per_1k')['mean']; l1=cs['table1']['1.0']['mean']; l0=cs['table1']['0.0']['mean']
    rec=100*(g-l1)/(l0-l1)
    gs=E(p,k,'graft1','markers_per_1k')['seeds']; l1s=cs['table1']['1.0']['seeds']; l0s=cs['table1']['0.0']['seeds']
    recs=st.mean(100*(a-b)/(c_-b) for a,b,c_ in zip(gs,l1s,l0s))
    c.chk(f'{p}/{k} recovered%',v[5][0][0],0,rec,f'recomputed from means (per-seed mean {recs:.1f})')
    print(f'  {p}/{k} recovered {rec:.2f}% (per-seed {recs:.2f}); length ratio {E(p,k,"graft1","avg_tokens")["mean"]/cs["table2"]["tokens1"]["mean"]:.2f}; other>shared {E(p,k,"graft_other1","markers_per_1k")["mean"]>g}; whole>shared {E(p,k,"graft_whole1","markers_per_1k")["mean"]>g}')
c.report()
# tab09
c=Checker('Table 9 middle (tab09 block 0)')
for p,s,r in cellrows('tab09.tex',0):
    k=sname(s); v=[tok(x) for x in r[2:]]; cs=ss[f'{p}/{k}']
    ms(c,f'{p}/{k} l0 head',v[0],cs['base']['base0_head_markers_per_1k'],1,'base0_head')
    ms(c,f'{p}/{k} l1 head',v[1],cs['base']['base1_head_markers_per_1k'],1,'base1_head')
    for i,e in enumerate(['graft2','graft_other2','graft_whole2']):
        ms(c,f'{p}/{k} {e} head',v[2+i],E(p,k,e,'head_markers_per_1k'),1,e+' head')
    g=E(p,k,'graft2','head_markers_per_1k')['mean']; l0=cs['base']['base0_head_markers_per_1k']['mean']
    print(f'  {p}/{k} graft2 head/l0 head={100*g/l0:.0f}%  whole2 head={E(p,k,"graft_whole2","head_markers_per_1k")["mean"]:.2f} vs l1 head {cs["base"]["base1_head_markers_per_1k"]["mean"]:.2f}')
# tab09 block 1 (completion/accuracy incl. beta=1 and Whole) is checked in t_new.py
c.report()
# tab10
c=Checker('Table 10 top (tab10 block 0)')
for p,s,r in cellrows('tab10.tex'):
    k=sname(s); v=[tok(x) for x in r[2:]]; cs=ss[f'{p}/{k}']
    ms(c,f'{p}/{k} l0',v[0],cs['table1']['0.0'],1,'table1 0.0')
    for i,e in enumerate(['ampn1','ampn4','ampV']): ms(c,f'{p}/{k} {e}',v[1+i],E(p,k,e,'markers_per_1k'),1,e)
    ms(c,f'{p}/{k} tok0',v[4][:2],cs['table2']['tokens0'],1,'tokens0')
    ms(c,f'{p}/{k} ampV tok',v[4][2:],E(p,k,'ampV','avg_tokens'),1,'ampV tokens')
    l0=cs['table1']['0.0']['mean']
    print(f'  {p}/{k} amp ratios: g1 {E(p,k,"ampn1","markers_per_1k")["mean"]/l0:.3f} g4 {E(p,k,"ampn4","markers_per_1k")["mean"]/l0:.3f} V {E(p,k,"ampV","markers_per_1k")["mean"]/l0:.3f}; tok ratio V {E(p,k,"ampV","avg_tokens")["mean"]/cs["table2"]["tokens0"]["mean"]:.2f}')
c.report()
