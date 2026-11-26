"""
Recompute the LLM accuracies of Fernandez-Pichel et al. (npj Digital Medicine, 2025)
from the answer files saved in their repository, and compare with the stored values.

Usage:
  python reproduction/check_fernandez_pichel.py . reproduction/recompute_scores.json

Columns: uniq = unique questions, noans = answers with no yes/no, mine = recomputed
accuracy, stored = accuracy line written by the authors. Files produced with an older
prompt ("The answer must be Yes or No") are flagged DIFF because that sentence is
matched first; trust the stored value for those.
"""
import re, sys, os, json
import xml.etree.ElementTree as ET
ROOT = sys.argv[1]
LLM = os.path.join(ROOT, 'src', 'llm')

def load_topics(year):
    root = ET.parse(os.path.join(LLM, 'evaluation', f'misinfo-resources-{year}', 'topics', f'misinfo-{year}-topics.xml')).getroot()
    out = []
    for t in root.findall('topic'):
        if year == 2022:
            q = t.find('question').text; a = t.find('answer').text
        elif year == 2021:
            q = t.find('description').text; a = 'yes' if t.find('stance').text == 'helpful' else 'no'
        else:
            q = t.find('description').text; a = t.find('answer').text
        out.append((q.rstrip(), a.strip().lower()))
    return out

MARKERS = ['We are a committee', 'We are a non-expert', 'I am a non-expert', 'Q: Will wearing an ankle brace',
           'You are a helpful medical assistant', '<s>[INST]', 'Accuracy:']

def score(path, year):
    topics = load_topics(year)
    seen = {}
    for q, a in topics:
        seen[q] = a
    uniq = list(seen.items())
    txt = open(path, encoding='utf-8', errors='replace').read()
    pos = []
    cur = 0
    for q, a in uniq:
        i = txt.find(q, cur)
        pos.append(i)
        if i >= 0:
            cur = i + len(q)
    hits = 0; found = 0; noans = 0
    for k, (q, a) in enumerate(uniq):
        i = pos[k]
        if i < 0:
            continue
        found += 1
        start = i + len(q)
        nxt = [p for p in pos[k+1:] if p >= 0]
        end = nxt[0] if nxt else len(txt)
        seg = txt[start:end]
        cut = len(seg)
        for m in MARKERS:
            j = seg.find(m)
            if j >= 0:
                cut = min(cut, j)
        seg = seg[:cut].lower()
        m = re.search(r'\byes\b|\bno\b', seg)
        if not m:
            noans += 1
        elif m.group() == a:
            hits += 1
    stored = re.findall(r'Accuracy:\s*([0-9.]+)', txt)
    return dict(n_topics=len(topics), n_unique=len(uniq), found=found, hits=hits, noans=noans,
                acc=round(hits/len(uniq), 4), stored=(float(stored[-1]) if stored else None))

rows = []
for mode in ['zero-shot', 'few-shot']:
    base = os.path.join(LLM, 'outputs', mode)
    for model in sorted(os.listdir(base)):
        for f in sorted(os.listdir(os.path.join(base, model))):
            ym = re.search(r'(20(?:20|21|22))', f)
            if not ym: continue
            year = int(ym.group(1))
            prompt = 'expert' if f.startswith('expert') else ('nonexpert' if f.startswith('nonexpert') else 'nocontext')
            shot = '0'
            sm = re.search(r'_(\d)(?:\.txt)?$', f) or re.search(r'\.txt_(\d)$', f)
            if mode == 'few-shot' and sm: shot = sm.group(1)
            r = score(os.path.join(base, model, f), year)
            r.update(mode=mode, model=model, file=f, year=year, prompt=prompt, shot=shot, verbose=('verbose' in f))
            rows.append(r)
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
print(f"{'mode':9} {'model':17} {'file':26} {'uniq':>4} {'found':>5} {'noans':>5} {'hits':>4} {'mine':>6} {'stored':>7}  flag")
for r in rows:
    flag = ''
    if r['stored'] is None: flag = 'NO-STORED'
    elif abs(r['stored'] - r['acc']) > 0.005: flag = 'DIFF'
    print(f"{r['mode']:9} {r['model']:17} {r['file']:26} {r['n_unique']:>4} {r['found']:>5} {r['noans']:>5} {r['hits']:>4} {r['acc']:>6} {str(r['stored']):>7}  {flag}")
