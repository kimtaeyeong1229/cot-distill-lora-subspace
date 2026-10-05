import re, json, statistics as st
TAB='/home/lab/workspace/NAACL_cot_distill_lora_subspace/tables/'
EXP='/home/lab/workspace/cot-distill-lora-bit/experiments/'
def J(name): return json.load(open(EXP+name+'.json'))
PAIRS=['A','B','C']; STUD=['Qwen','Llama','OLMo']
FULL={'Qwen':'Qwen2.5-0.5B','Llama':'Llama-3.2-1B','OLMo':'OLMo-2-1B'}
XS={'Qwen':'Qwen-0.5B','Llama':'Llama-1B','OLMo':'OLMo-1B'}
def P(p,alt='D'): return alt if p=='C' else p
NUM=re.compile(r'[-+]?\d[\d,]*\.?\d*')
def parse_num_tokens(cell):
    c=cell.replace('$-$','-').replace('$+$','+').replace('$\\pm$',' PM ').replace('$\\rightarrow$',' TO ')
    c=re.sub(r'\\[a-zA-Z]+\{[^}]*\}','',c)  # drop commands like \multicolumn{..}
    toks=[]
    for m in re.finditer(r'(?<![\w.])([-+]?)(\d[\d,]*(?:\.\d+)?)',c):
        s=m.group(2).replace(',','')
        dec=len(s.split('.')[1]) if '.' in s else 0
        v=float(s)*(-1 if m.group(1)=='-' else 1)
        toks.append((v,dec))
    return toks
def rows(fname, block=0):
    txt=open(TAB+fname).read()
    blocks=re.findall(r'\\midrule(.*?)\\bottomrule',txt,re.S)
    # take content between first \midrule after header and \bottomrule for each tabular
    tabs=re.split(r'\\begin\{tabular\}\{',txt)[1:]
    out=[]
    t=tabs[block]
    body=t.split('\\toprule',1)[1]
    # header ends at first \midrule
    hdr,body=body.split('\\midrule',1)
    body=body.split('\\bottomrule')[0]
    body=body.replace('\\midrule','')
    for line in body.split('\\\\'):
        line=line.strip()
        if not line: continue
        cells=[x.strip() for x in line.split('&')]
        out.append(cells)
    return out
class Checker:
    def __init__(s,name): s.name=name; s.n=0; s.err=[]; s.border=[]
    def chk(s,label,shown,dec,val,src=''):
        """shown: displayed float, dec decimals, val: JSON value already scaled"""
        s.n+=1
        u=10**(-dec); d=abs(val-shown)
        if d>0.5*u+1e-9:
            s.err.append(f'{label}: LaTeX {shown:.{dec}f} vs JSON {val:.{dec+3}f} ({src})')
        elif d>=0.49*u:
            s.border.append(f'{label}: LaTeX {shown:.{dec}f} vs JSON {val:.{dec+4}f} ({src})')
    def report(s):
        print(f'### {s.name}: {s.n} values checked, {len(s.err)} mismatches, {len(s.border)} borderline')
        for e in s.err: print('  MISMATCH', e)
        for e in s.border: print('  borderline', e)
def tok(cell,i=None):
    t=parse_num_tokens(cell)
    return t if i is None else t[i]
def cellrows(fname,block=0):
    """yield (pair, student, cells[2:]) with pair carried forward"""
    cur=None
    for r in rows(fname,block):
        if r[0].strip(): cur=r[0].strip()
        stu=r[1].strip() if len(r)>1 else ''
        yield cur, stu, r
def sname(s):
    for k,v in FULL.items():
        if s.startswith(v): return k
    return s
QLO=[f'{p}/{k}' for p in PAIRS for k in STUD]   # column order of the transposed appendix tables
def trows(fname):
    """Rows of a transposed table (one header; sections titled by a \\multicolumn row): yields (section, label, cells)."""
    txt=open(TAB+fname).read()
    tab=re.split(r'\\begin\{tabular\}\{',txt)[1]
    body=re.split(r'(?<!c)\\midrule',tab,1)[1].split('\\bottomrule')[0]
    sec=''
    for line in body.split('\\\\'):
        line=re.sub(r'(?<!c)\\midrule','',line).strip()
        if not line: continue
        if line.startswith('\\multicolumn'):
            sec=re.search(r'\\emph\{(.*)\}\}',line).group(1); continue
        cells=[x.strip() for x in line.split('&')]
        yield sec, cells[0], cells[1:]
APP_TEX='/home/lab/workspace/NAACL_cot_distill_lora_subspace/sections/09_appendix.tex'
def caption(label):
    s=open(APP_TEX).read(); i=s.index('\\label{'+label+'}'); j=s.rindex('\\caption{',0,i); return s[j:i]
