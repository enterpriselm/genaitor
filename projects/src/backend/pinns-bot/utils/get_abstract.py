import json
import requests
import fitz
import os

papers_path = 'src/papers_urls.json'
with open(papers_path, 'r') as f:
    data = json.loads(f.readlines()[0])

urls = data['urls']

for url in urls:
    if 'arxiv' in url:
        try:
            request = requests.get(url).content
            filename = url.split('pdf/')[1].replace('.','_')
            with open('src/'+filename+'.pdf', 'wb') as f:
                f.write(request)
        except:
            pass

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
    return text

def extract_abstract(text):
    import re
    match = re.search(r'(?i)(abstract)(.*?)(introduction|1\.)', text, re.DOTALL)
    if match:
        return match.group(2).strip()
    else:
        return None

pdfs = ['src/'+x for x in os.listdir('src') if x.endswith('pdf')]
for pdf in pdfs:
    try:
        text = extract_text_from_pdf(pdf)
        abstract = extract_abstract(text)
        
        
        if abstract:
            data = {
            "path": pdf,
            "abstract": abstract
            }
            with open('src/abstract_data.json', 'a') as f:
                f.write(json.dumps(data)) 
    except:
        pass