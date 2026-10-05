from common import *
ss=J('seed_spread'); bl=J('baselines'); sae=J('sae_llama'); ood=J('ood'); bs=J('base_students')
def ms(c,label,toks,d,scale=1,src=''):
    c.chk(label+' mean',toks[0][0],toks[0][1],scale*d['mean'],src)
    c.chk(label+' sd',toks[1][0],toks[1][1],scale*d['sd'],src)
# tab11
c=Checker('tab11')
keys0=['base0.0','base1.0','lambda0.0_rm_pre','lambda0.0_rm_decisive']
for r in rows('tab11.tex',0):
    p=r[0].strip(); v=[tok(x) for x in r[1:]]
    for i,k in enumerate(keys0): ms(c,f'{p} {k}',v[i],sae[p][k]['sae'],1,f'sae_llama.{k}.sae')
keys1=[('graft2','sae'),('graft_orth2','sae'),('ampV','sae'),('base0.0','sae_masked')]
for r in rows('tab11.tex',1):
    p=r[0].strip(); v=[tok(x) for x in r[1:]]
    for i,(k,f) in enumerate(keys1): ms(c,f'{p} {k}.{f}',v[i],sae[p][k][f],1,f'sae_llama.{k}.{f}')
c.report()
for p in PAIRS:
    b0=sae[p]['base0.0']['sae']['mean']; s=sae[p]
    print(f'  {p}: corr_keyword_sae {s["corr_keyword_sae"]:.3f}; masked/base0 {s["base0.0"]["sae_masked"]["mean"]/b0:.3f}; rm_pre/b0 {s["lambda0.0_rm_pre"]["sae"]["mean"]/b0:.3f}; rm_other/b0 {s["lambda0.0_rm_decisive"]["sae"]["mean"]/b0:.3f}; ampn1 {s["ampn1"]["sae"]["mean"]/b0:.3f} ampn4 {s["ampn4"]["sae"]["mean"]/b0:.3f} ampV {s["ampV"]["sae"]["mean"]/b0:.3f} (sd {s["ampV"]["sae"]["sd"]:.3f}); graft2 {s["graft2"]["sae"]["mean"]:.3f} vs base1 {s["base1.0"]["sae"]["mean"]:.3f}; rm_scale/b0 {s["lambda0.0_rm_scale"]["sae"]["mean"]/b0:.3f}')
    k0=s['base0.0']['keyword']['mean']
    print(f'     keyword ratios: rm_other {s["lambda0.0_rm_decisive"]["keyword"]["mean"]/k0:.3f} ampn1 {s["ampn1"]["keyword"]["mean"]/k0:.3f} ampn4 {s["ampn4"]["keyword"]["mean"]/k0:.3f} ampV {s["ampV"]["keyword"]["mean"]/k0:.3f}; graft_orth2 head? ')
# tab12
c=Checker('tab12')
toks={b:{k:[] for k in ('base0','rm_pre','rm_other')} for b in ('gsm8k','gpqa_diamond')}
acc={b:{k:[] for k in ('base0','rm_pre','rm_other')} for b in ('gsm8k','gpqa_diamond')}
ratios={b:{'rm_pre':[],'rm_other':[],'tok':[]} for b in ('gsm8k','gpqa_diamond')}
mrows=[]
for p,s,r in cellrows('tab12.tex'):
    if p.startswith('\\multicolumn'): mrows.append([tok(x)[0] for x in r[1:]]); continue
    k=sname(s); v=[tok(x) for x in r[2:]]
    for bi,b in enumerate(('gsm8k','gpqa_diamond')):
        o=ood[b][f'{p}/{k}']
        for j,e in enumerate(('base0','rm_pre','rm_other')):
            ms(c,f'{b} {p}/{k} {e}',v[3*bi+j],o[e]['markers'],1,f'ood.{b}.{e}.markers')
            toks[b][e].append(o[e]['tokens']['mean']); acc[b][e].append(100*o[e]['accuracy']['mean'])
        ratios[b]['rm_pre'].append(100*o['rm_pre']['markers']['mean']/o['base0']['markers']['mean'])
        ratios[b]['rm_other'].append(100*o['rm_other']['markers']['mean']/o['base0']['markers']['mean'])
        ratios[b]['tok'].append(o['base0']['tokens']['mean']/o['rm_pre']['tokens']['mean'])
for bi,b in enumerate(('gsm8k','gpqa_diamond')):
    for j,e in enumerate(('base0','rm_pre','rm_other')):
        c.chk(f'Mean tokens {b} {e}',mrows[0][3*bi+j][0],0,st.mean(toks[b][e]),'nine-cell mean')
        c.chk(f'Mean acc {b} {e}',mrows[1][3*bi+j][0],1,st.mean(acc[b][e]),'nine-cell mean')
c.report()
for b in ratios: print(f'  {b}: rm_pre kept% {min(ratios[b]["rm_pre"]):.2f}-{max(ratios[b]["rm_pre"]):.2f}; rm_other kept% {min(ratios[b]["rm_other"]):.1f}-{max(ratios[b]["rm_other"]):.1f}; token ratio {min(ratios[b]["tok"]):.2f}-{max(ratios[b]["tok"]):.2f}')
print('  per-cell GPQA acc max per cond:',{e:round(max(acc['gpqa_diamond'][e]),1) for e in acc['gpqa_diamond']}, 'base1 max', round(max(100*ood['gpqa_diamond'][f'{p}/{k}']['base1']['accuracy']['mean'] for p in PAIRS for k in STUD),1))
# tab13
c=Checker('tab13')
col={e:[] for e in ('base0','rm_pre','rm_other','base1','tr0','tr1')}
for p,s,r in cellrows('tab13.tex'):
    if p.startswith('\\multicolumn'): mr=[tok(x) for x in r[1:]]; continue
    k=sname(s); v=[tok(x) for x in r[2:]]; o=ood['gsm8k'][f'{p}/{k}']
    for j,e in enumerate(('base0','rm_pre','rm_other','base1')):
        ms(c,f'{p}/{k} acc {e}',v[j],o[e]['accuracy'],100,f'ood.gsm8k.{e}.accuracy'); col[e].append(100*o[e]['accuracy']['mean'])
    ms(c,f'{p}/{k} trunc base0',v[4][:2],o['base0']['truncated'],100,'base0.truncated')
    ms(c,f'{p}/{k} trunc rm_pre',v[4][2:],o['rm_pre']['truncated'],100,'rm_pre.truncated')
    col['tr0'].append(100*o['base0']['truncated']['mean']); col['tr1'].append(100*o['rm_pre']['truncated']['mean'])
for j,e in enumerate(('base0','rm_pre','rm_other','base1')): c.chk(f'Mean {e}',mr[j][0][0],1,st.mean(col[e]),'nine-cell mean')
c.chk('Mean trunc0',mr[4][0][0],0,st.mean(col['tr0']),'nine-cell mean'); c.chk('Mean trunc rm_pre',mr[4][1][0],0,st.mean(col['tr1']),'nine-cell mean')
c.report()
print('  gsm8k trunc0 range',round(min(col['tr0']),2),round(max(col['tr0']),2),' rm_pre trunc max',round(max(col['tr1']),2))
print('  rm_pre - rm_other acc per cell',[round(a-b,2) for a,b in zip(col['rm_pre'],col['rm_other'])])
print('  rm_pre - base0 acc per cell',[round(a-b,2) for a,b in zip(col['rm_pre'],col['base0'])])
print('  rm_pre - base1 acc per cell',[round(a-b,2) for a,b in zip(col['rm_pre'],col['base1'])])
# tab14
c=Checker('tab14')
conds=['base0','rmseed','budget','rmseed_other','gs_tau','nothink']
agg={e:{'acc':[],'tok':[],'mk':[]} for e in conds}
mr=[]
for p,s,r in cellrows('tab14.tex',0):
    if p.startswith('\\multicolumn'): mr.append([tok(x)[0] for x in r[1:]]); continue
    k=sname(s); v=[tok(x) for x in r[2:]]; b=bl[f'{p}/{k}']
    for j,e in enumerate(conds):
        ms(c,f'{p}/{k} acc {e}',v[j],b[e]['accuracy'],100,f'baselines.{e}.accuracy')
        agg[e]['acc'].append(100*b[e]['accuracy']['mean']); agg[e]['tok'].append(b[e]['tokens']['mean']); agg[e]['mk'].append(b[e]['markers']['mean'])
for j,e in enumerate(conds):
    c.chk(f'Mean acc {e}',mr[0][j][0],1,st.mean(agg[e]['acc']),'nine-cell mean')
    c.chk(f'Mean tokens {e}',mr[1][j][0],0,st.mean(agg[e]['tok']),'nine-cell mean')
    c.chk(f'Mean markers {e}',mr[2][j][0],1,st.mean(agg[e]['mk']),'nine-cell mean')
print('  rmseed_other tokens / base0 tokens %:',[round(100*a/b) for a,b in zip(agg['rmseed_other']['tok'],agg['base0']['tok'])])
SFULL={'Qwen':'Qwen2.5-0.5B-Instruct','Llama':'Llama-3.2-1B-Instruct','OLMo':'OLMo-2-1B-Instruct'}
agg2={e:{'acc':[],'tok':[],'mk':[],'mk2':[]} for e in ('merge0.5','base1','untrained')}
mr=[]
for p,s,r in cellrows('tab14.tex',1):
    if p.startswith('\\multicolumn'): mr.append([tok(x)[0] for x in r[1:]]); continue
    k=sname(s); v=[tok(x) for x in r[2:]]; b=bl[f'{p}/{k}']; u=bs[SFULL[k]]
    ms(c,f'{p}/{k} merge0.5',v[0],b['merge0.5']['accuracy'],100,'baselines.merge0.5.accuracy')
    ms(c,f'{p}/{k} base1',v[1],b['base1']['accuracy'],100,'baselines.base1.accuracy')
    ms(c,f'{p}/{k} untrained',v[2],u['accuracy'],100,'base_students.accuracy')
    for e,src in (('merge0.5',b['merge0.5']),('base1',b['base1'])):
        agg2[e]['acc'].append(100*src['accuracy']['mean']); agg2[e]['tok'].append(src['tokens']['mean']); agg2[e]['mk'].append(src['markers']['mean']); agg2[e]['mk2'].append(src['markers']['mean'])
    agg2['untrained']['acc'].append(100*u['accuracy']['mean']); agg2['untrained']['tok'].append(u['avg_tokens']['mean']); agg2['untrained']['mk'].append(u['markers_first_sample']['mean']); agg2['untrained']['mk2'].append(u['reflective_per_1k_words']['mean'])
for j,e in enumerate(('merge0.5','base1','untrained')):
    c.chk(f'Mean acc {e}',mr[0][j][0],1,st.mean(agg2[e]['acc']),'nine-cell mean')
    c.chk(f'Mean tokens {e}',mr[1][j][0],0,st.mean(agg2[e]['tok']),'nine-cell mean')
    c.chk(f'Mean markers {e}',mr[2][j][0],1,st.mean(agg2[e]['mk']),'nine-cell mean (markers_first_sample for untrained)')
print('  untrained markers alt (reflective_per_1k_words):',round(st.mean(agg2['untrained']['mk2']),3))
print('  base1 markers mean from seed_spread table1 1.0:', round(st.mean(ss[f'{p}/{k}']['table1']['1.0']['mean'] for p in PAIRS for k in STUD),3))
c.report()
# tab15 (Table 1) is checked in t_new.py
print('  rm_pre markers range', round(min(bl[f'{p}/{k}']['rm_pre']['markers']['mean'] for p in PAIRS for k in STUD),4), round(max(bl[f'{p}/{k}']['rm_pre']['markers']['mean'] for p in PAIRS for k in STUD),4))
print('  rm_pre markers seed_spread B/OLMo', ss['B/OLMo']['edits']['rm_pre']['markers_per_1k']['mean'])
print('  GPQA per-cell acc >25%:',[(p,k,e,round(100*ood['gpqa_diamond'][f'{p}/{k}'][e]['accuracy']['mean'],1)) for p in PAIRS for k in STUD for e in ('base0','rm_pre','rm_other','base1') if 100*ood['gpqa_diamond'][f'{p}/{k}'][e]['accuracy']['mean']>25])
