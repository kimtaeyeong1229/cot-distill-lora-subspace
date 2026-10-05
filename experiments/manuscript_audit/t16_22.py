from common import *
ss=J('seed_spread'); bl=J('baselines'); to=J('top_overlap'); dg=J('degeneration'); es=J('edit_stats'); th=J('threshold_sensitivity')
ph=J('phi_resample_s012'); px=J('phi_xseed'); 
# tab16
c=Checker('tab16')
for p,s,r in cellrows('tab16.tex'):
    k=sname(s); v=[tok(x) for x in r[2:]]; t=to[f'{p}/{k}']; b=bl[f'{p}/{k}']
    c.chk(f'{p}/{k} dims_top',v[0][0][0],1,t['dims_top'],'top_overlap.dims_top')
    c.chk(f'{p}/{k} cover_region',v[1][0][0],3,t['cover_region'],'cover_region'); c.chk(f'{p}/{k} cover_region sd',v[1][1][0],3,t['cover_region_sd'],'cover_region_sd')
    c.chk(f'{p}/{k} cover_top',v[2][0][0],3,t['cover_top'],'cover_top')
    c.chk(f'{p}/{k} chance',v[3][0][0],4,t['chance'],'chance')
    b0=b['base0']['markers']['mean']
    kr=100*b['rm_pre']['markers']['mean']/b0; kt=100*b['gs_energy']['markers']['mean']/b0
    krs=st.mean(100*a/z for a,z in zip(b['rm_pre']['markers']['seeds'],b['base0']['markers']['seeds']))
    c.chk(f'{p}/{k} kept shared%',v[4][0][0],1,kr,f'rm_pre/base0 markers (per-seed mean {krs:.2f}; edit_stats {es["cells"][f"{p}/{k}"]["rm_retained"]["value"]:.2f})')
    c.chk(f'{p}/{k} kept top%',v[5][0][0],1,kt,'gs_energy/base0 markers')
    c.chk(f'{p}/{k} tok base0',v[6][0][0],0,b['base0']['tokens']['mean'],'baselines.base0.tokens')
    c.chk(f'{p}/{k} tok rm_pre',v[7][0][0],0,b['rm_pre']['tokens']['mean'],'baselines.rm_pre.tokens')
    c.chk(f'{p}/{k} tok gs_energy',v[8][0][0],0,b['gs_energy']['tokens']['mean'],'baselines.gs_energy.tokens')
c.report()
# tab17
c=Checker('tab17')
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
c=Checker('tab20')
ratios=[]
for p,s,r in cellrows('tab20.tex'):
    k=sname(s); v=[tok(x)[0] for x in r[2:]]; t=th[f'{p}/{k}']
    for i,cut in enumerate(('0.4','0.5','0.6')):
        sr=t['shared_region'][cut]
        c.chk(f'{p}/{k} cut{cut} dims',v[3*i][0],1,sr['dims'],'dims'); c.chk(f'{p}/{k} cut{cut} pre',v[3*i+1][0],3,sr['pre_loo'],'pre_loo'); c.chk(f'{p}/{k} cut{cut} post',v[3*i+2][0],3,sr['post'],'post')
        ratios.append(sr['pre_loo']/sr['post'])
    for i,cut in enumerate(('0.65','0.75','0.85')): c.chk(f'{p}/{k} two-seed {cut}',v[9+i][0],0,100*t['two_seed_region'][cut]['overlap'],'two_seed_region.overlap')
c.report()
print(f'  pre/post ratio range {min(ratios):.2f}-{max(ratios):.2f}')
# tab21
c=Checker('tab21')
for p,s,r in cellrows('tab21.tex'):
    k=sname(s); v=[tok(x) for x in r[2:]]; rr=ph[f'{P(p)}/{FULL[k]}']['rows']; t=th[f'{p}/{k}']; x=px[f'{P(p)}/{XS[k]}']
    for i,l in enumerate(['0.0','0.2','0.4','0.6','0.8','1.0']): c.chk(f'{p}/{k} B({l})',v[i][0][0],3,rr[l]['baseline'],'phi_resample_s012 rows.baseline')
    c.chk(f'{p}/{k} xseed phi',v[6][0][0],3,x['0.0']['xseed'],'phi_xseed 0.0.xseed'); c.chk(f'{p}/{k} r*phi',v[6][1][0],1,16*x['0.0']['xseed'],'16*xseed')
    for j,f in enumerate(('within_pre','within_post','across')): c.chk(f'{p}/{k} {f}',v[7+j][0][0],3,t['cross_seed_phi'][f],'cross_seed_phi')
    c.chk(f'{p}/{k} chance',v[10][0][0],4,t['chance'],'threshold_sensitivity.chance')
    print(f'  {p}/{k} within_post/chance {t["cross_seed_phi"]["within_post"]/t["chance"]:.1f}  post>across {t["cross_seed_phi"]["within_post"]>t["cross_seed_phi"]["across"]}')
c.report()
# tab22 (Table 18, tab:length) is checked in t22_length.py
