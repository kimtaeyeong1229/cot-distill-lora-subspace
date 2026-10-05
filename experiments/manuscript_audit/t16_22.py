from common import *
ss=J('seed_spread'); bl=J('baselines'); to=J('top_overlap'); dg=J('degeneration'); es=J('edit_stats'); th=J('threshold_sensitivity')
ph=J('phi_resample_s012'); px=J('phi_xseed'); 
# old Table 19 (tab16): its top-direction columns moved into tab23 (Table 4, checked in t_new.py); the token range
# in the Table 4 caption is checked in text_checks.py
# tab17 (Table 13, transposed loops and beta 2/4/8) is checked in t06_10.py
# tab19 (Table 22) is checked in t_new.py
# tab20
c=Checker('Table 5 bottom (tab20)')
ratios=[]
for r in rows('tab20.tex'):
    cell=r[0]; v=[tok(x)[0] for x in r[1:]]; t=th[cell]
    for i,cut in enumerate(('0.4','0.6')):   # cut 0.5 is printed once, in tab05 (checked in t01_05.py); two-seed overlap is in tab05
        sr=t['shared_region'][cut]
        c.chk(f'{cell} cut{cut} dims',v[3*i][0],1,sr['dims'],'dims'); c.chk(f'{cell} cut{cut} pre',v[3*i+1][0],3,sr['pre_loo'],'pre_loo'); c.chk(f'{cell} cut{cut} post',v[3*i+2][0],3,sr['post'],'post')
    for cut in ('0.4','0.5','0.6'): ratios.append(t['shared_region'][cut]['pre_loo']/t['shared_region'][cut]['post'])
c.report()
print(f'  pre/post ratio range {min(ratios):.2f}-{max(ratios):.2f}')
# old Table 16 (tab21): the A/Qwen B(lambda) collapse is App B text, checked in t01_05.py; the cross-seed overlap sentence was cut
# tab22 (Table 18, tab:length) is checked in t22_length.py
