#!/usr/bin/env python3
"""Read-only basic manuscript-PDF preflight: no experimental data or code accessed.
Usage: python F29R7_Q1_PDF_Preflight.py '/path/to/current_manuscript.pdf'
Requires PyMuPDF: pip install pymupdf
"""
import sys, re
from pathlib import Path
import fitz

if len(sys.argv) != 2:
    raise SystemExit('Usage: python F29R7_Q1_PDF_Preflight.py manuscript.pdf')
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f'Missing PDF: {p}')
doc=fitz.open(p)
texts=[pg.get_text(sort=True) for pg in doc]
whole='\n'.join(texts)
print('F29R7 — Q1 MANUSCRIPT PDF PREFLIGHT (read-only)')
print('PDF:',p.name)
print('Pages:',len(doc))
print('\nPOSSIBLE FIGURE SPELLING OR REPETITION ISSUES')
flags={
 'Development typo':'devlopement',
 'Sensitivity typo':'sentitive',
 'Benjamini-Hochberg typo':'hockberg',
 'Internal protocol label':'Step-35',
 'Internal phase label':'Phase A metadata',
 'Redundant best-seed label':'No best-seed or model selection',
 'Author contributions spacing':'manuscript -writing',
}
for label,term in flags.items():
    pg=[str(i+1) for i,t in enumerate(texts) if term.lower() in t.lower()]
    print(f'{label:37} : {", ".join(pg) if pg else "none found"}')
print('\nFUNDING DECLARATION')
# Extract section as rendered; manual verification still required.
mm=re.search(r'(?im)^Funding\s*(.*?)^Ethics approval\b',whole,flags=re.S|re.M)
if mm:
    section=mm.group(1).strip()
    print('Empty' if not section else f'Nonempty: {section[:240]}')
else:
    print('Not confidently parsed; manual check required')
print('\nSMALL TEXT SPANS BY PAGE (counts, caution: may include figure annotations)')
for i,page in enumerate(doc):
    ss=[(span['size'],span['text'].strip())
        for block in page.get_text('dict')['blocks'] if 'lines' in block
        for line in block['lines'] for span in line['spans'] if span['text'].strip()]
    tiny=[x for x in ss if x[0] < 6.0]
    if tiny:
        example=min(tiny,key=lambda x:x[0]);print(f'page {i+1:2d}: {len(tiny):3d} spans <6pt, min {example[0]:.2f}pt, example {example[1][:45]}')
print('\nThis is a typography/text preflight, not a scientific validation.')
