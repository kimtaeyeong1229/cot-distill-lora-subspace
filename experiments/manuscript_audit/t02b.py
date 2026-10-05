from common import *
md=J('main_density'); LAM=['0.0','0.2','0.4','0.6','0.8','1.0']
c=Checker('Table 6 bottom (tab02 block 1)')
peak={}
for p,s,r in cellrows('tab02.tex',1):
    k=sname(s); d=md[f'{p}/{k}']
    for i,l in enumerate(LAM): c.chk(f'{p}/{k} acc {l}',tok(r[2+i])[0][0],1,100*d[l]['accuracy'],f'main_density.{p}/{k}.{l}.accuracy')
    acc={l:100*d[l]['accuracy'] for l in LAM}; m=max(acc.values()); peak[f'{p}/{k}']=[l for l in LAM if abs(acc[l]-m)<1e-9]
c.report()
print('  peak lambda per cell (exact):',peak)
bs=J('base_students'); U={'Qwen':100*bs['Qwen2.5-0.5B-Instruct']['accuracy']['mean'],'Llama':100*bs['Llama-3.2-1B-Instruct']['accuracy']['mean'],'OLMo':100*bs['OLMo-2-1B-Instruct']['accuracy']['mean']}
below=[(p,k,l) for p in PAIRS for k in STUD for l in LAM if 100*md[f'{p}/{k}'][l]['accuracy']<U[k]]
above=[(p,k,l) for p in PAIRS for k in STUD for l in LAM if 100*md[f'{p}/{k}'][l]['accuracy']>=U[k]]
print('  untrained',U,' below count',len(below),' above/equal:',above)
