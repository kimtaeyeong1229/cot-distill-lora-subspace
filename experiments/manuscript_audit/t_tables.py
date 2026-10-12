"""Every table value of the manuscript (body Tables 1-3, appendix Tables 4-17) and the numbers in their captions and in the
App. A sentences that replaced the cut-sensitivity and top-direction tables, against the experiment JSONs."""
import re, math
from common import *

ss = J('seed_spread'); bl = J('baselines'); me = J('main_edits'); rv = J('review_stats'); em = J('energy_match')
es = J('edit_stats'); pb = J('paired_bootstrap'); ra = J('region_anatomy'); to = J('top_overlap'); dg = J('degeneration')
lc = J('length_confound'); td = J('teacher_density'); md = J('main_density'); ru = J('removal_vs_untrained')
cp = J('consensus_pre'); cq = J('consensus_post'); th = J('threshold_sensitivity'); sae = J('sae_llama')
ood = J('ood'); bs = J('base_students'); ts = J('teacher_structure'); sst = J('student_structure')['summary']['all']['level']
a1 = J('alignment_s012'); a2 = J('alignment_markers_s012'); a3 = J('alignment_length_s012'); tc = J('think_rate_completed')
CELLS = QLO
E = lambda c, e, f: ss[c]['edits'][e][f]['mean']
TOTAL = [0, 0]
BODY4 = SECTIONS + '04_where_the_adapter_switches.tex'; APP = open(APP_TEX).read()


def done(c):
    c.report(); TOTAL[0] += c.n; TOTAL[1] += len(c.err)


def cap_of(path, label):
    s = open(path).read(); i = s.index('\\label{' + label + '}'); return s[s.rindex('\\caption{', 0, i):i]


def rng(c, label, text, d, vals, src):
    c.chk(label + ' lo', float(text[0]), d, min(vals), src); c.chk(label + ' hi', float(text[1]), d, max(vals), src)


# ---------------- Table 1 (tab31) ----------------
c = Checker('Table 1 (tab31)')
for r in rows('tab31.tex'):
    p = r[0].strip(); a, b = r[1].strip().split('--'); key = f'{float(a):.1f}->{float(b):.1f}'
    iv = [x for x in lc['pairs'][p]['intervals'] if x['interval'] == key][0]
    if r[2].strip().lower() != iv['class']: c.err.append(f'{p} {key} class {r[2]}')
    v = [tok(x)[0] for x in r[3:]]
    k = [iv['kept_vs_half_reference'][s] for s in STUD]
    c.chk(f'{p} {key} dlogL', v[0][0], 2, iv['log_len_drop'], 'log_len_drop')
    c.chk(f'{p} {key} dthink', v[1][0], 0, iv['think'][1] - iv['think'][0], 'think[1]-think[0]')
    c.chk(f'{p} {key} marker ratio', v[2][0], 2, iv['markers'][1] / iv['markers'][0], 'markers[1]/markers[0]')
    c.chk(f'{p} {key} kept mean', v[3][0], 2, st.mean(k), 'mean kept_vs_half_reference')
    c.chk(f'{p} {key} loss mean', v[4][0], 2, st.mean(-math.log(x) / iv['log_len_drop'] for x in k), 'mean -ln kept / dlogL')
cap = cap_of(BODY4, 'tab:lenbody')
cs = J('pairA_crossseed'); v = [x for k in ('A/Qwen', 'A/Llama', 'A/OLMo') for x in cs[k]['switch_over_held_by_seed']]
fl = J('two_seed_floor')['floors']; w = [x for f in ('chance', 'lambda1') for x in fl[f]['pairA_switch_over_held'].values()]
m = re.search(r'ratio is ([\d.]+)--([\d.]+) with a cross-seed build and ([\d.]+)--([\d.]+) with two floors', cap)
rng(c, 'caption cross-seed', m.groups()[:2], 2, v, 'pairA_crossseed'); rng(c, 'caption floors', m.groups()[2:], 2, w, 'two_seed_floor')
done(c)

# ---------------- Table 2 (tab32) ----------------
c = Checker('Table 2 (tab32)')
rec = lambda x, e, f: 100 * (me[x][e][f] - me[x]['base1'][f]) / (me[x]['base0'][f] - me[x]['base1'][f])
h = lambda x, e: ss[x]['edits'][e]['head_markers_per_1k']['mean'] / ss[x]['base']['base0_head_markers_per_1k']['mean']
mk = lambda x, k: bl[x][k]['markers']['mean']
spec = {('Markers kept', 'Region'): lambda x: 100 * mk(x, 'rm_pre') / mk(x, 'base0'),
        ('Markers kept', 'Other directions'): lambda x: 100 * mk(x, 'rm_other') / mk(x, 'base0'),
        ('Markers kept', 'Top components'): lambda x: 100 * mk(x, 'gs_energy') / mk(x, 'base0'),
        ('Markers kept', 'Random directions'): lambda x: 100 * mk(x, 'rm_spanrand') / mk(x, 'base0'),
        ('Accuracy gain', 'Region removed'): lambda x: es['cells'][x]['rm_minus_cut']['value'],
        ('Early-marker gap', 'Region, $\\beta=1$'): lambda x: 100 * rv['graft_recovery_early'][x]['beta1'],
        ('Early-marker gap', 'Region, $\\beta=2$'): lambda x: 100 * rv['graft_recovery_early'][x]['beta2'],
        ('Early-marker gap', 'Whole update, $\\beta=2$'): lambda x: rec(x, 'graft_full2', 'head_markers_per_1k'),
        ('Whole-response gap', 'Region, $\\beta=2$'): lambda x: rec(x, 'graft2', 'markers_per_1k'),
        ('Whole-response gap', 'Whole update, $\\beta=2$'): lambda x: rec(x, 'graft_full2', 'markers_per_1k'),
        ('Completed', '$\\lambda=1$ host'): lambda x: 100 * (1 - ss[x]['table2']['trunc1']['mean']),
        ('Completed', 'Region graft, $\\beta=2$'): lambda x: 100 * ss[x]['edits']['graft2']['completed']['mean']}
seen = set()
for sec, lab, cells in trows('tab32.tex'):
    key = [k for k in spec if sec.startswith(k[0]) and k[1] == lab]
    if not key: c.err.append(f'no rule [{sec}] {lab}'); continue
    seen.add(key[0])
    for j, p in enumerate(PAIRS):
        t = tok(cells[j])[0]; c.chk(f'{lab} {p}', t[0], t[1], st.mean(spec[key[0]](f'{p}/{k}') for k in STUD), 'pair mean')
for k in spec:
    if k not in seen: c.err.append(f'row missing {k}')
done(c)

# ---------------- Table 3 (tab15) ----------------
c = Checker('Table 3 (tab15)')
m9 = lambda f: st.mean(f(p, k) for p in PAIRS for k in STUD)
for row in rows('tab15.tex', 0):
    key = {'$\\lambda=0$': 'base0', 'Region removed': 'rm_pre', 'Other removed': 'rm_other', '$\\lambda=1$': 'base1'}[row[0].strip()]
    v = [tok(x)[0] for x in row[1:]]
    for bi, b in enumerate(('gsm8k', 'gpqa_diamond')):
        c.chk(f'{key} {b} tokens', v[2 * bi][0], 0, m9(lambda p, k: ood[b][f'{p}/{k}'][key]['tokens']['mean']), 'ood tokens')
        c.chk(f'{key} {b} acc', v[2 * bi + 1][0], 1, m9(lambda p, k: 100 * ood[b][f'{p}/{k}'][key]['accuracy']['mean']), 'ood accuracy')
SF = {'Qwen': 'Qwen2.5-0.5B-Instruct', 'Llama': 'Llama-3.2-1B-Instruct', 'OLMo': 'OLMo-2-1B-Instruct'}
spec3 = {'$\\lambda=0$': 'base0', 'Region removed': 'rm_pre', 'Two-seed region removed': 'rmseed', 'Gold-Switch-style removal ($\\tau$ rank)': 'gs_tau',
         'Length-matched cut': 'budget', 'Think-skip (adapted)': 'nothink', 'Other removed (two-seed region)': 'rmseed_other',
         'Merged with $\\lambda=1$': 'merge0.5', '$\\lambda=1$': 'base1', 'Untrained': None}
for row in rows('tab15.tex', 1):
    name = row[0].strip(); v = [tok(x)[0] for x in row[1:]]; key = spec3[name]
    if key:
        f = [lambda p, k, q=q: bl[f'{p}/{k}'][key][q]['mean'] for q in ('markers', 'tokens', 'accuracy')]
    else:
        f = [lambda p, k: bs[SF[k]]['markers_first_sample']['mean'], lambda p, k: bs[SF[k]]['avg_tokens']['mean'], lambda p, k: bs[SF[k]]['accuracy']['mean']]
    c.chk(f'{name} markers', v[0][0], 1, m9(f[0]), 'markers'); c.chk(f'{name} tokens', v[1][0], 0, m9(f[1]), 'tokens')
    c.chk(f'{name} acc', v[2][0], 1, 100 * m9(f[2]), 'accuracy')
done(c)

# ---------------- Table 4 (tab23) and 5 (tab24) ----------------
c = Checker('Tables 4 and 8 (tab23, tab24)')
for r in rows('tab23.tex'):
    cell = r[0].strip(); a = ra[cell]; v = [tok(x)[0] for x in r[1:]]
    c.chk(f'{cell} dims', v[0][0], 1, a['overall']['dims'], 'dims')
    c.chk(f'{cell} share0', v[1][0], 0, 100 * a['overall']['share_0.0'], 'share_0.0'); c.chk(f'{cell} share1', v[2][0], 0, 100 * a['overall']['share_1.0'], 'share_1.0')
    c.chk(f'{cell} chance', v[3][0], 1, 100 * a['overall']['chance_share'], 'chance_share')
    for j, dpt in enumerate(('early', 'middle', 'late')): c.chk(f'{cell} {dpt}', v[4 + j][0], 0, 100 * a['by_depth'][dpt]['share_0.0'], 'by_depth')
for r in rows('tab24.tex'):
    cell = r[0].strip(); mm = em[cell]['mean']; v = [tok(x)[0] for x in r[1:]]
    for j, (f, sc, d) in enumerate([('region_B', 1, 2), ('ratio_B', 1, 2), ('ratio_dW', 1, 2), ('unreachable', 100, 0), ('dims_region', 1, 1), ('dims_other', 1, 1)]):
        c.chk(f'{cell} {f}', v[j][0], d, sc * mm[f], f'energy_match.mean.{f}')
done(c)

# ---------------- Table 10 (tab02, tab02b) ----------------
c = Checker('Table 10 (tab02, tab02b)')
for r in rows('tab02.tex'):
    cell = r[0].strip(); t2 = ss[cell]['table2']; v = [tok(x)[0] for x in r[1:]]; p, k = cell.split('/')
    c.chk(f'{cell} tok0', v[0][0], 0, t2['tokens0']['mean'], 'tokens0'); c.chk(f'{cell} tok1', v[1][0], 0, t2['tokens1']['mean'], 'tokens1')
    c.chk(f'{cell} tr0', v[2][0], 0, 100 * t2['trunc0']['mean'], 'trunc0'); c.chk(f'{cell} tr1', v[3][0], 0, 100 * t2['trunc1']['mean'], 'trunc1')
    c.chk(f'{cell} dacc', v[4][0], 1, t2['d_acc']['mean'], 'd_acc'); c.chk(f'{cell} daccC', v[5][0], 1, t2['d_acc_completed']['mean'], 'd_acc_completed')
    c.chk(f'{cell} r', v[6][0], 2, td[p]['corr'][k], 'teacher_density.corr')
for r in rows('tab02b.tex'):
    cell = r[0].strip(); v = [tok(x)[0] for x in r[1:]]
    for i, l in enumerate(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']): c.chk(f'{cell} acc {l}', v[i][0], 1, 100 * md[cell][l]['accuracy'], 'main_density')
m = re.search(r'\(([\d.]+) over all 54 points\)', caption('tab:2')); c.chk('pooled r', float(m.group(1)), 2, td['pooled_corr'], 'pooled_corr')
done(c)

# ---------------- Table 12 (tab04) ----------------
c = Checker('Table 12 (tab04)')
sig = None
for r in rows('tab04.tex'):
    if r[0].strip(): sig = r[0].strip()
    pr = r[1].strip(); v = [tok(x) for x in r[2:]]
    if sig.startswith(('Completed', 'Training')):
        a = tc['alignment']['completed' if sig.startswith('Completed') else 'train']; cells_, summ = a['cells'], a['summary']
    else:
        a = a1 if sig.startswith('Think') else (a2 if sig.startswith('Markers') else a3); cells_, summ = a, a['_summary']
    if pr == 'Pooled':
        c.chk(f'{sig} pooled mean', v[3][0][0], v[3][0][1], summ['pooled']['mean_r'], 'pooled.mean_r')
        c.chk(f'{sig} pooled p', v[4][0][0], v[4][0][1], summ['pooled']['p'], 'pooled.p'); continue
    for j, k in enumerate(STUD): c.chk(f'{sig} {pr}/{k} r', v[j][0][0], 2, cells_[f'{P(pr)}/{FULL[k]}']['r'], 'r')
    c.chk(f'{sig} {pr} mean', v[3][0][0], 2, summ[P(pr)]['mean_r'], 'mean_r'); c.chk(f'{sig} {pr} p', v[4][0][0], v[4][0][1], summ[P(pr)]['p'], 'p')
done(c)

# ---------------- Table 16 (tab05) + caption cuts ----------------
c = Checker('Table 16 (tab05)')
key = lambda cell: f"{P(cell[0])}/{FULL[cell.split('/')[1]]}"
for r in rows('tab05.tex'):
    cell = r[0].strip(); e = cp[key(cell)]; q = cq[key(cell)]; v = [tok(x)[0] for x in r[1:]]
    c.chk(f'{cell} switch', v[0][0], 1, float(e['transition']), 'transition'); c.chk(f'{cell} dims pre', v[1][0], 1, e['dims'], 'dims')
    c.chk(f'{cell} dims post', v[2][0], 1, q['dims'], 'post dims'); c.chk(f'{cell} in_loo', v[3][0], 3, e['in_loo'], 'in_loo')
    c.chk(f'{cell} out', v[4][0], 3, e['out'], 'out'); c.chk(f'{cell} chance', v[5][0], 3, e['chance'], 'chance')
    c.chk(f'{cell} two-seed 0.75', v[6][0], 0, 100 * th[cell]['two_seed_region']['0.75']['overlap'], 'two_seed 0.75')
m = re.search(r'\((\d+)--(\d+)\\% at 0.65, (\d+)--(\d+)\\% at 0.85\)', caption('tab:5'))
for i, cut in enumerate(('0.65', '0.85')): rng(c, f'caption two-seed {cut}', m.groups()[2 * i:2 * i + 2], 0, [100 * th[x]['two_seed_region'][cut]['overlap'] for x in CELLS], 'two_seed')
done(c)

# ---------------- Table 9 (tab22) ----------------
c = Checker('Table 9 (tab22)')
cur = None
for r in rows('tab22.tex'):
    if r[0].strip(): cur = r[0].strip()
    a, b = r[1].strip().split('--'); k_ = f'{float(a):.1f}->{float(b):.1f}'
    iv = [x for x in lc['pairs'][cur]['intervals'] if x['interval'] == k_][0]
    if r[2].strip().lower() != iv['class']: c.err.append(f'{cur} {k_} class')
    v = [tok(x)[0] for x in r[3:]]
    c.chk(f'{cur} {k_} dlogL', v[0][0], 2, iv['log_len_drop'], 'log_len_drop'); c.chk(f'{cur} {k_} think end', v[1][0], 0, iv['think'][1], 'think[1]')
    for j, s in enumerate(STUD): c.chk(f'{cur} {k_} kept {s}', v[2 + j][0], 2, iv['kept_vs_half_reference'][s], 'kept')
    for j, f in enumerate(['coverage_change', 'leave_lambda_out_coverage_change']): c.chk(f'{cur} {k_} {f}', v[5 + j][0], 2, st.mean(iv[f].values()), f'mean {f}')
m = re.search(r'\((\d+)\\%, (\d+)\\% and (\d+)\\% at \$\\lambda=0\$\)', caption('tab:length'))
for x, p in zip(m.groups(), PAIRS): c.chk(f'caption think0 {p}', float(x), 0, lc['pairs'][p]['intervals'][0]['think'][0], 'think[0]')
done(c)

# ---------------- Table 7 (tab26) ----------------
c = Checker('Table 7 (tab26)')
kap = lambda d: d['vs_untrained']['response_level']['cohen_kappa']; un = lambda x: ru['untrained'][x.split('/')[1]]
for r in rows('tab26.tex'):
    x = r[0].strip(); v = [tok(t)[0] for t in r[1:]]; cc = ru['cells'][x]
    vals = [100 * cc['rm_pre']['accuracy'], 100 * un(x)['accuracy'], cc['rm_pre']['tokens'], un(x)['tokens'], 100 * cc['rm_pre']['opens_okay'],
            kap(cc['rm_pre']), kap(cc['lambda1']), un(x)['self_ceiling']['cohen_kappa']]
    for j, (val, nm) in enumerate(zip(vals, ['acc rem', 'acc untr', 'tok rem', 'tok untr', 'okay', 'kappa rem', 'kappa l1', 'kappa self'])):
        c.chk(f'{x} {nm}', v[j][0], v[j][1], val, 'removal_vs_untrained')
done(c)

# ---------------- Table 6 (tab27) ----------------
c = Checker('Table 6 (tab27)')
colmap = [('A', '0.0'), ('A', '0.2'), ('A', '0.4'), ('B', '0.0'), ('B', '0.4'), ('C', '0.0'), ('C', '0.4')]
fld = {'Markers': (lambda d: d['a_markers_per_1k_all_generations'], 1), 'Words': (lambda d: d['a_mean_words'], 0), 'Tag (\\%)': (lambda d: 100 * d['b_think_tag'], 0),
       'Recap (\\%)': (lambda d: 100 * d['b_tagged_recap'], 0), 'Opener (\\%)': (lambda d: 100 * d['e_opens_monologue'], 0),
       'Sent. (\\%)': (lambda d: 100 * d['e_monologue_sentence_share'], 0), 'Verif.': (lambda d: d['c_markers_per_1k']['verification'], 2),
       'Check (\\%)': (lambda d: 100 * d['d_any_verification'], 0), 'Hesit.': (lambda d: d['c_markers_per_1k']['hesitation'], 2),
       'Altern.': (lambda d: d['c_markers_per_1k']['alternative'], 2), 'Attempts': (lambda d: d['c_attempts_per_solution'], 2)}
for r in rows('tab27.tex'):
    name = r[0].strip()
    if name not in fld: c.err.append(f'no rule {name}'); continue
    fn, d = fld[name]
    for j, (p, l) in enumerate(colmap): c.chk(f'{p} {l} {name}', tok(r[1 + j])[0][0], d, fn(ts[p]['per_lambda'][l]), 'teacher_structure')
done(c)


# ---------------- grid tables (tab06, tab08, tab10) ----------------
def grid(c, fname, spec):
    seen = set()
    for sec, lab, cells in trows(fname):
        key = sorted([k for k in spec if sec.startswith(k[0]) and k[1] == lab], key=lambda k: -len(k[0]))   # longest section prefix wins
        if not key: c.err.append(f'{fname}: no rule [{sec}] {lab}'); continue
        seen.add(key[0]); fn, sc = spec[key[0]]
        for cell, t in zip(CELLS, [tok(x)[0] for x in cells[:9]]): c.chk(f'[{sec[:12]}] {lab} {cell}', t[0], t[1], sc * fn(cell), fname)
    for k in spec:
        if k not in seen: c.err.append(f'{fname}: row missing {k}')
    done(c)


c = Checker('Table 5 (tab10, amplification)')
spec = {('Whole', '$\\lambda=0$'): (lambda x: ss[x]['table1']['0.0']['mean'], 1), ('Early', '$\\lambda=0$'): (lambda x: ss[x]['base']['base0_head_markers_per_1k']['mean'], 1)}
for lab, e in (('$\\gamma=1$', 'ampn1'), ('$\\gamma=4$', 'ampn4'), ('All in region', 'ampV')):
    spec[('Whole', lab)] = ((lambda e: lambda x: E(x, e, 'markers_per_1k'))(e), 1)
    spec[('Early', lab)] = ((lambda e: lambda x: E(x, e, 'head_markers_per_1k'))(e), 1)
for lab, g in (('$\\gamma=1$', 'gain_ampn1'), ('$\\gamma=4$', 'gain_ampn4'), ('All in region', 'gain_ampV')):
    spec[('Region weight', lab)] = ((lambda g: lambda x: em[x]['mean'][g])(g), 1)
grid(c, 'tab10.tex', spec)
cap = caption('tab:10')
m = re.search(r'change ([\d.]+)--([\d.]+)\$\\times\$ at \$\\gamma \\le 4\$ and ([\d.]+)--([\d.]+)\$\\times\$ with all in region', cap)
c = Checker('Table 5 caption')
r4 = [E(x, e, 'head_markers_per_1k') / ss[x]['base']['base0_head_markers_per_1k']['mean'] for x in CELLS for e in ('ampn1', 'ampn4')]
rV = [E(x, 'ampV', 'head_markers_per_1k') / ss[x]['base']['base0_head_markers_per_1k']['mean'] for x in CELLS]
rng(c, 'early gamma<=4', m.groups()[:2], 2, r4, 'edits.ampn1/ampn4 head / base0 head'); rng(c, 'early all', m.groups()[2:], 2, rV, 'edits.ampV head / base0 head')
m = re.search(r'accuracy falls from ([\d.]+)\\% to ([\d.]+)--([\d.]+)\\% and completion from (\d+)\\% to (\d+)--(\d+)\\%', cap)
nc = rv['amplification']['nine_cell']
c.chk('acc l0', float(m.group(1)), 1, 100 * nc['base0']['accuracy'], 'amplification.nine_cell')
rng(c, 'acc amp', m.groups()[1:3], 1, [100 * nc[e]['accuracy'] for e in ('ampn1', 'ampn4', 'ampV')], 'amplification.nine_cell')
c.chk('done l0', float(m.group(4)), 0, 100 * nc['base0']['completed'], 'amplification.nine_cell')
rng(c, 'done amp', m.groups()[4:6], 0, [100 * nc[e]['completed'] for e in ('ampn1', 'ampn4', 'ampV')], 'amplification.nine_cell')
done(c)

c = Checker('Table 11 (tab06, removal)')
spec = {('Whole', '$\\lambda=0$'): (lambda x: ss[x]['table1']['0.0']['mean'], 1),
        ('Whole', 'Region removed'): (lambda x: E(x, 'rm_pre', 'markers_per_1k'), 1),
        ('Whole', 'Length-matched cut'): (lambda x: bl[x]['budget']['markers']['mean'], 1),
        ('Whole', 'Other removed'): (lambda x: E(x, 'rm_other', 'markers_per_1k'), 1),
        ('Whole', 'Top removed'): (lambda x: bl[x]['gs_energy']['markers']['mean'], 1),
        ('Whole', 'Scaled down'): (lambda x: E(x, 'rm_scale', 'markers_per_1k'), 1),
        ('Whole', 'Random removed'): (lambda x: bl[x]['rm_spanrand']['markers']['mean'], 1),
        ('Whole', 'Post-switch removed'): (lambda x: bl[x]['rm_post']['markers']['mean'], 1)}
for lab, cn in (('$\\lambda=0$', 'base0'), ('Region removed', 'lambda0.0_rm_pre'), ('Length-matched cut', 'lambda0.0_budget'), ('Other removed', 'lambda0.0_rm_decisive'),
                ('Top removed', 'lambda0.0_gs_energy'), ('Scaled down', 'lambda0.0_rm_scale'),
                ('Random removed', 'lambda0.0_rm_spanrand'), ('Post-switch removed', 'lambda0.0_rm_post')):
    spec[('Early', lab)] = ((lambda cn: lambda x: me[x][cn]['head_markers_per_1k'])(cn), 1)
accf = {'$\\lambda=0$': lambda x: ss[x]['base']['base0_accuracy']['mean'], 'Region removed': lambda x: E(x, 'rm_pre', 'accuracy'),
        'Length-matched cut': lambda x: bl[x]['budget']['accuracy']['mean'], 'Other removed': lambda x: E(x, 'rm_other', 'accuracy'),
        'Top removed': lambda x: bl[x]['gs_energy']['accuracy']['mean'], 'Scaled down': lambda x: E(x, 'rm_scale', 'accuracy'),
        'Random removed': lambda x: bl[x]['rm_spanrand']['accuracy']['mean'], 'Post-switch removed': lambda x: bl[x]['rm_post']['accuracy']['mean']}
for lab, f in accf.items(): spec[('MATH', lab)] = (f, 100)
grid(c, 'tab06.tex', spec)
c = Checker('Table 11 caption')
m = re.search(r'ine-cell mean accuracy (?:is )?([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+)\\%', caption('tab:6'))
for x, (lab, f) in zip(m.groups(), accf.items()): c.chk(f'mean acc {lab}', float(x), 1, 100 * st.mean(f(y) for y in CELLS), 'nine-cell mean')
m = re.search(r'seeds of ([\d.]+), ([\d.]+) and (\d+\.\d)', caption('tab:6'))
for i, x in enumerate(m.groups()): c.chk(f'B/Llama seed {i}', float(x), 1, ss['B/Llama']['table1']['0.0']['seeds'][i], 'table1 seeds')
done(c)

c = Checker('Table 17 (tab08, grafts and loops)')
def rec1(x):
    g = E(x, 'graft1', 'markers_per_1k'); l1 = ss[x]['table1']['1.0']['mean']; l0 = ss[x]['table1']['0.0']['mean']; return 100 * (g - l1) / (l0 - l1)
spec = {('Early', '$\\lambda=1$'): (lambda x: ss[x]['base']['base1_head_markers_per_1k']['mean'], 1),
        ('Early', 'Region'): (lambda x: E(x, 'graft2', 'head_markers_per_1k'), 1), ('Early', 'Other'): (lambda x: E(x, 'graft_other2', 'head_markers_per_1k'), 1),
        ('Early', 'Whole'): (lambda x: E(x, 'graft_whole2', 'head_markers_per_1k'), 1),
        ('Whole-response', '$\\lambda=1$'): (lambda x: ss[x]['table1']['1.0']['mean'], 1),
        ('Whole-response', 'Region, $\\beta=1$'): (lambda x: E(x, 'graft1', 'markers_per_1k'), 1),
        ('Whole-response', 'Other, $\\beta=1$'): (lambda x: E(x, 'graft_other1', 'markers_per_1k'), 1),
        ('Whole-response', 'Whole, $\\beta=1$'): (lambda x: E(x, 'graft_whole1', 'markers_per_1k'), 1),
        ('Whole-response', 'Region, $\\beta=2$'): (lambda x: E(x, 'graft2', 'markers_per_1k'), 1),
        ('Whole-response', 'Other, $\\beta=2$'): (lambda x: E(x, 'graft_other2', 'markers_per_1k'), 1),
        ('Whole-response', 'Whole, $\\beta=2$'): (lambda x: E(x, 'graft_whole2', 'markers_per_1k'), 1),
        ('Completed', 'Region, $\\beta=1$'): (lambda x: E(x, 'graft1', 'completed'), 100),
        ('Completed', 'Region, $\\beta=2$'): (lambda x: E(x, 'graft2', 'completed'), 100),
        ('Completed', 'Other, $\\beta=2$'): (lambda x: E(x, 'graft_other2', 'completed'), 100),
        ('Completed', 'Whole, $\\beta=2$'): (lambda x: E(x, 'graft_whole2', 'completed'), 100),
        ('MATH', 'Region'): (lambda x: E(x, 'graft2', 'accuracy'), 100), ('MATH', 'Other'): (lambda x: E(x, 'graft_other2', 'accuracy'), 100),
        ('MATH', 'Whole'): (lambda x: E(x, 'graft_whole2', 'accuracy'), 100),
        ('Tokens', 'Region, $\\beta=1$'): (lambda x: E(x, 'graft1', 'avg_tokens') / 1000, 1),
        ('Whole-response gap', 'Region, $\\beta=1$'): (rec1, 1)}
for lab, e in (('$\\lambda=0$', 'base0'), ('$\\lambda=1$', 'base1'), ('Region removed', 'rm_pre'), ('Region, $\\beta=1$', 'graft1'),
               ('Region, $\\beta=2$', 'graft2'), ('Other, $\\beta=2$', 'graft_other2'), ('Whole, $\\beta=2$', 'graft_whole2')):
    spec[('Responses', lab)] = ((lambda e: lambda x: dg[x][e]['mean'])(e), 100)
grid(c, 'tab08.tex', spec)

# ---------------- Table 15 (tab19, tab19b): value +- SD ----------------
c = Checker('Table 15 (tab19, tab19b)')
def pmcheck(lab, cell, val, seeds, src):
    t = tok(cell); c.chk(lab, t[0][0], t[0][1], val, src); c.chk(lab + ' sd', t[1][0], t[1][1], st.stdev(seeds), src + ' seeds SD')
f0 = ['rm_retained', 'rm_acc', 'rm_minus_top', 'rm_minus_cut']; pbk = {'rm_minus_top': 'top', 'rm_minus_cut': 'cut'}
for r in rows('tab19.tex'):
    lab = r[0].strip()
    if lab in es['cells']:
        for j, f in enumerate(f0): pmcheck(f'{lab} {f}', r[1 + j], es['cells'][lab][f]['value'], es['cells'][lab][f]['seeds'], f'edit_stats.{f}')
        continue
    for j, f in enumerate(f0):
        t = tok(r[1 + j])
        if not t: continue
        n = es['nine_cell'][f]; d = t[0][1]
        if lab == 'Mean': c.chk(f'mean {f}', t[0][0], d, n['mean'], 'nine_cell.mean')
        elif 'cells fixed' in lab:
            ci = pb['nine_cell'][pbk[f]]['ci95_pp'] if f in pbk else n['ci95']; c.chk(f'ci lo {f}', t[0][0], d, ci[0], 'ci'); c.chk(f'ci hi {f}', t[1][0], d, ci[1], 'ci')
        elif 'cells as units' in lab:
            if f in pbk: ci = rv['cell_interval'][pbk[f]]['ci95']
            else:
                v_ = [es['cells'][x][f]['value'] for x in CELLS]; h_ = 2.306 * st.stdev(v_) / math.sqrt(9); ci = [st.mean(v_) - h_, st.mean(v_) + h_]
            c.chk(f'cell ci lo {f}', t[0][0], d, ci[0], 'cell interval'); c.chk(f'cell ci hi {f}', t[1][0], d, ci[1], 'cell interval')
        elif lab == 'Median': c.chk(f'median {f}', t[0][0], d, n['median'], 'median')
        elif 'w/o' in lab: c.chk(f'w/o {f}', t[0][0], d, n['mean_without_B_C_Qwen'], 'w/o')
        else: c.err.append(f'unknown row {lab}')
for r in rows('tab19b.tex'):
    lab = r[0].strip()
    if lab in es['cells']:
        for j, f in enumerate(['graft2_head', 'graft2_done']): pmcheck(f'{lab} {f}', r[1 + j], es['cells'][lab][f]['value'], es['cells'][lab][f]['seeds'], f'edit_stats.{f}')
        for j, b in enumerate(['beta1', 'beta2']):
            pmcheck(f'{lab} rec {b}', r[3 + j], 100 * rv['graft_recovery_early'][lab][b], [100 * s for s in rv['graft_seed_recovery'][lab][b]['seeds']], 'graft recovery')
        continue
    for j, f in enumerate(['graft2_head', 'graft2_done']):
        t = tok(r[1 + j]); n = es['nine_cell'][f]; d = t[0][1]
        if lab == 'Mean': c.chk(f'mean {f}', t[0][0], d, n['mean'], 'mean')
        elif 'cells fixed' in lab: c.chk(f'ci lo {f}', t[0][0], d, n['ci95'][0], 'ci'); c.chk(f'ci hi {f}', t[1][0], d, n['ci95'][1], 'ci')
        elif lab == 'Median': c.chk(f'median {f}', t[0][0], d, n['median'], 'median')
        elif 'w/o' in lab: c.chk(f'w/o {f}', t[0][0], d, n['mean_without_B_C_Qwen'], 'w/o')
        else: c.err.append(f'unknown row {lab}')
done(c)

# ---------------- Table 13 (tab29, tab11) ----------------
c = Checker('Table 13 (tab29, tab11)')
conds = ['lambda0', 'rm_pre', 'rm_decisive', 'graft2', 'lambda1', 'untrained']
rowf = {'Hesitation': lambda h, w: h['c_markers_per_1k']['hesitation'], 'Verification': lambda h, w: h['c_markers_per_1k']['verification'],
        'Attempts': lambda h, w: h['c_attempts_per_solution'], 'Opener': lambda h, w: h['e_opens_monologue'],
        'Mono.\\ sent.': lambda h, w: h['e_monologue_sentence_share'], 'Recap (compl.)': lambda h, w: w.get('b_boxed_recap_completed')}
for r in rows('tab29.tex'):
    name = r[0].strip()
    for j, cn in enumerate(conds):
        cell = r[1 + j].strip()
        if cell in ('--', '-'): continue
        t = tok(cell)[0]; c.chk(f'{name} {cn}', t[0], t[1], rowf[name](sst[cn]['head'], sst[cn]['whole']), 'student_structure')
rowkey = {'$\\lambda=0$': ('base0.0', 'sae'), '$\\lambda=1$': ('base1.0', 'sae'), 'Region removed': ('lambda0.0_rm_pre', 'sae'),
          'Other removed': ('lambda0.0_rm_decisive', 'sae'), 'Graft region ($\\beta=2$)': ('graft2', 'sae'), 'Graft other ($\\beta=2$)': ('graft_orth2', 'sae'), 'Graft whole update ($\\beta=2$)': ('graft_full2', 'sae'),
          'All in region': ('ampV', 'sae'), '$\\lambda=0$, markers deleted': ('base0.0', 'sae_masked')}
for r in rows('tab11.tex'):
    name = r[0].strip(); v = [tok(x)[0] for x in r[1:]]
    if name.startswith('$r$'):
        for i, p in enumerate(PAIRS): c.chk(f'{p} r', v[i][0], 2, sae[p]['corr_keyword_sae'], 'corr_keyword_sae')
        continue
    k, f = rowkey[name]
    for i, p in enumerate(PAIRS): c.chk(f'{p} {k}.{f}', v[i][0], v[i][1], sae[p][k][f]['mean'], 'sae_llama')
done(c)

# ---------------- App A sentences that replaced the cut-sensitivity and top-direction tables ----------------
c = Checker('App A (former cut-sensitivity and top-direction tables)')
m = re.search(r'Rebuilt at cuts 0.4 and 0.6, the region has ([\d.]+)--([\d.]+) and ([\d.]+)--([\d.]+) directions\. Pre-switch coverage stays ([\d.]+)--([\d.]+) times', APP)
g = m.groups()
for i, cut in enumerate(('0.4', '0.6')): rng(c, f'cut {cut} dims', g[2 * i:2 * i + 2], 1, [th[x]['shared_region'][cut]['dims'] for x in CELLS], 'threshold_sensitivity dims')
ratios = [th[x]['shared_region'][cut]['pre_loo'] / th[x]['shared_region'][cut]['post'] for x in CELLS for cut in ('0.4', '0.5', '0.6')]
rng(c, 'pre/post ratio', g[4:6], 1, ratios, 'pre_loo/post')
m = re.search(r'mostly inside it \((\d+)--(\d+)\\%, against at most ([\d.]+)\\% by chance\), and they span (\d+)--(\d+)\\% of it', APP)
g = m.groups()
rng(c, 'top inside %', g[0:2], 0, [100 * to[x]['cover_top'] for x in CELLS], 'top_overlap.cover_top')
c.chk('top chance max %', float(g[2]), 1, 100 * max(to[x]['chance'] for x in CELLS), 'top_overlap.chance')
rng(c, 'region spanned %', g[3:5], 0, [100 * to[x]['cover_region'] for x in CELLS], 'top_overlap.cover_region')
done(c)

# ---------------- Table 14 (tab25, examples) ----------------
ex = J('examples')
c = Checker('Table 14 (tab25)')
for r, e in zip(rows('tab25.tex'), ex['students'] + ex['teachers']):
    r[0] = r[0].replace(r'\addlinespace', '').strip()
    if r[0] != e['row']: c.err.append(f'row name {r[0]} vs {e["row"]}')
    if 'status' in e and r[4].strip() != e['status']: c.err.append(f'{e["row"]} status {r[4]} vs {e["status"]}')
    c.chk(f'{e["row"]} tokens', tok(r[2])[0][0], 0, e['tokens'], 'examples.tokens'); c.chk(f'{e["row"]} markers', tok(r[3])[0][0], 1, e['markers_per_1k'], 'examples.markers')
done(c)
print(f'\nTOTAL: {TOTAL[0]} table/caption values checked, {TOTAL[1]} mismatches')
