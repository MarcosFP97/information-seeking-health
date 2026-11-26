"""
Compare our rerun answers with the authors' saved answers, question by question.

Usage (from the repo root):
  python reproduction/compare_answers.py <authors_file> <our_file> <year>
"""
import re, sys, os
import xml.etree.ElementTree as ET

def load_topics(year):
    p = os.path.join('src', 'llm', 'evaluation', f'misinfo-resources-{year}', 'topics', f'misinfo-{year}-topics.xml')
    out = {}
    for t in ET.parse(p).getroot().findall('topic'):
        if year == 2022:
            q = t.find('question').text; a = t.find('answer').text
        elif year == 2021:
            q = t.find('description').text; a = 'yes' if t.find('stance').text == 'helpful' else 'no'
        else:
            q = t.find('description').text; a = t.find('answer').text
        out[q.rstrip()] = a.strip().lower()
    return list(out.items())

MARKERS = ['We are a committee', 'We are a non-expert', 'I am a non-expert', 'You are a helpful medical assistant', 'Accuracy:']

def answers(path, topics):
    txt = open(path, encoding='utf-8', errors='replace').read()
    pos, cur = [], 0
    for q, _ in topics:
        i = txt.find(q, cur); pos.append(i)
        if i >= 0: cur = i + len(q)
    res = []
    for k, (q, _) in enumerate(topics):
        i = pos[k]
        if i < 0: res.append('MISSING'); continue
        nxt = [p for p in pos[k+1:] if p >= 0]
        seg = txt[i + len(q): nxt[0] if nxt else len(txt)]
        cut = min([seg.find(m) for m in MARKERS if seg.find(m) >= 0] or [len(seg)])
        m = re.search(r'\byes\b|\bno\b', seg[:cut].lower())
        res.append(m.group() if m else 'NONE')
    return res

authors, ours, year = sys.argv[1], sys.argv[2], int(sys.argv[3])
topics = load_topics(year)
a, o = answers(authors, topics), answers(ours, topics)
same = sum(x == y for x, y in zip(a, o))
acc_a = sum(x == t[1] for x, t in zip(a, topics)) / len(topics)
acc_o = sum(x == t[1] for x, t in zip(o, topics)) / len(topics)
print(f"questions: {len(topics)} | identical answers: {same} | different: {len(topics) - same}")
print(f"accuracy authors: {acc_a:.2f} | accuracy ours: {acc_o:.2f}")
for k, (x, y) in enumerate(zip(a, o)):
    if x != y:
        q, gt = topics[k]
        print(f"  Q{k+1}: authors={x} ours={y} correct={gt} | {q[:80]}")
