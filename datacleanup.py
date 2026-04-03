import pandas as pd
import re

def clean_resume(text):
    if not isinstance(text, str): return ""
    # Remove placeholders like [EMAIL], [PHONE]
    text = re.sub(r'\[.*?\]', '', text) 
    # Collapse multiple newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load specifically handling potential quoting issues in dirty CSVs
df = pd.read_csv('Candidates and Jobs.xlsx - candidate dump 1.csv', skipinitialspace=True)
df['clean_text'] = df['Resume Text'].apply(clean_resume)

# Extract "Name" (usually the first non-empty line)
def get_name(text):
    lines = [l.strip() for l in str(text).split('\n') if l.strip()]
    return lines[0] if lines else "Unknown"

df['candidate_name'] = df['Resume Text'].apply(get_name)
