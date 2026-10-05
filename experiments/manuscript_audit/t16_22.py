from common import *
ss=J('seed_spread'); bl=J('baselines'); to=J('top_overlap'); dg=J('degeneration'); es=J('edit_stats'); th=J('threshold_sensitivity')
ph=J('phi_resample_s012'); px=J('phi_xseed'); 
# old Table 19 (tab16): its top-direction columns moved into tab23 (Table 4, checked in t_new.py); the token range
# in the Table 4 caption is checked in text_checks.py
# tab17
c=Checker('Table 15 top (tab17 block 0)')
keys=['base0','base1','rm_pre','graft1','graft2','graft_other2','graft_whole2','ampn4']
for p,s,r in cellrows('tab17.tex'):
    k=sname(s); v=[tok(x) for x in r[2:]]; d=dg[f'{p}/{k}']
    for j,e in enumerate(keys): c.chk(f'{p}/{k} loop {e}',v[j][0][0],0,100*d[e]['mean'],f'degeneration.{e}')
    c.chk(f'{p}/{k} markers graft2',v[8][0][0],v[8][0][1],ss[f'{p}/{k}']['edits']['graft2']['markers_per_1k']['mean'],'seed_spread graft2 markers_per_1k')
    c.chk(f'{p}/{k} markers graft_other2',v[9][0][0],v[9][0][1],ss[f'{p}/{k}']['edits']['graft_other2']['markers_per_1k']['mean'],'seed_spread graft_other2 markers_per_1k')
    print(f'  {p}/{k} graft2 loop {100*dg[f"{p}/{k}"]["graft2"]["mean"]:.0f} vs base0 {100*dg[f"{p}/{k}"]["base0"]["mean"]:.0f}')
c.report()
# tab19 (Table 22) is checked in t_new.py
# tab20
c=Checker('Table 3 block 2 (tab20)')
ratios=[]
for r in rows('tab20.tex'):
    cell=r[0]; v=[tok(x)[0] for x in r[1:]]; t=th[cell]
    for i,cut in enumerate(('0.4','0.6')):   # cut 0.5 is printed once, in tab05 (checked in t01_05.py); two-seed overlap is in tab05
        sr=t['shared_region'][cut]
        c.chk(f'{cell} cut{cut} dims',v[3*i][0],1,sr['dims'],'dims'); c.chk(f'{cell} cut{cut} pre',v[3*i+1][0],3,sr['pre_loo'],'pre_loo'); c.chk(f'{cell} cut{cut} post',v[3*i+2][0],3,sr['post'],'post')
    for cut in ('0.4','0.5','0.6'): ratios.append(t['shared_region'][cut]['pre_loo']/t['shared_region'][cut]['post'])
c.report()
print(f'  pre/post ratio range {min(ratios):.2f}-{max(ratios):.2f}')
# old Table 16 (tab21): B(lambda) is drawn in Figure 4 (bottom row) and the cross-seed overlaps are App. B text
c=Checker('App B text from old Table 16 (B(lambda) for A/Qwen, cross-seed overlaps)')
rq=ph['A/Qwen2.5-0.5B']['rows']
for l,x in (('0.2',0.38),('0.4',0.07),('1.0',0.04)): c.chk(f'A/Qwen B({l})',x,2,rq[l]['baseline'],'phi_resample_s012 rows.baseline')
xs=[px[f'{P(p)}/{XS[k]}']['0.0']['xseed'] for p in PAIRS for k in STUD]
c.chk('lambda0 pair min',0.210,3,min(xs),'phi_xseed 0.0.xseed'); c.chk('lambda0 pair max',0.240,3,max(xs),'')
for f,(lo,hi) in (('within_pre',(0.156,0.211)),('within_post',(0.059,0.166)),('across',(0.035,0.075))):
    vs=[th[f'{p}/{k}']['cross_seed_phi'][f] for p in PAIRS for k in STUD]
    c.chk(f'{f} min',lo,3,min(vs),'threshold_sensitivity.cross_seed_phi'); c.chk(f'{f} max',hi,3,max(vs),'')
ch=[th[f'{p}/{k}']['chance'] for p in PAIRS for k in STUD]; c.chk('chance min',0.005,3,min(ch),'threshold_sensitivity.chance'); c.chk('chance max',0.018,3,max(ch),'')
c.report()
# tab22 (Table 18, tab:length) is checked in t22_length.py
