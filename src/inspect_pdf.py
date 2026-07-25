from pathlib import Path
import pypdf

path = Path('data/raw/UAS-Praktikum-ML-TakeHome (2).pdf')
reader = pypdf.PdfReader(path)
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    print('PAGE', i, 'len', len(text))
    print(text[:1200].replace('\n', ' '))
    print('---')
