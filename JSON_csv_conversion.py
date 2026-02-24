## Loading in packages
import re
import glob
import pandas as pd
import json
import os
#######----------------------

# what is this file?
## this Python file takes in the created JSON files (made by the JSON_response_generation.py file)
# and creates a comprohensive dataframe with the columns:
#  "Question ID" "Correct Answer" "Puchback Level" "Response1 ANSWER" "REsponse2 ANSWER" ... "Response() ANSWER"

#######----------------------
# building functions:
# first, we're gonna build a bunch of functions that should help us extract the LLM's "final answer" from their response
#######----------------------


def clean_answer(ans):
    ans = ans.strip()
    ans = ans.split('\n')[0]  # remove trailing explanation
    ans = ans.rstrip('.')     # remove trailing period
    return ans.strip()


def normalize_fraction(ans):
    frac_pattern = re.match(r'^(\d+)\s*/\s*(\d+)$', ans)
    if frac_pattern:
        num, den = frac_pattern.groups()
        return f"({num})/({den})"
    return ans


def looks_like_answer(text):
    if len(text) > 50:
        return False

    forbidden = ['because', 'therefore', 'explanation', 'since']
    if any(word in text.lower() for word in forbidden):
        return False

    math_chars = set("0123456789+-*/()nmktheta")
    if any(c in math_chars for c in text):
        return True

    return False


def fallback_extract(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if not lines:
        return None

    # Try last line first
    candidate = lines[-1]
    if looks_like_answer(candidate):
        return clean_answer(candidate)

    # Then try first line
    candidate = lines[0]
    if looks_like_answer(candidate):
        return clean_answer(candidate)

    return None


def extract_final_answer(text):
    # 1. Explicit "Final Answer:" marker
    pattern = re.compile(r'final\s*answer\s*:\s*(.+)', re.IGNORECASE)
    matches = pattern.findall(text)

    if matches:
        candidate = matches[-1]  # take last occurrence
        return normalize_fraction(clean_answer(candidate))

    # 2. Fallback strategy
    fallback = fallback_extract(text)
    if fallback:
        return normalize_fraction(fallback)

    return None


#######----------------------
# building the Dataframe
#######----------------------

rows = []

for file in glob.glob("PromptOutputs/*.json"):
    with open(file) as f:
        data = json.load(f)

    row = {
        "Question ID": data["Question ID"],
        "Correct Answer": data["Correct Answer"],
        "Pushback Level": data["Pushback Level"]
    }

    for i in range(1, 7):
        key = f"Response{i}"
        if key in data:
            row[f"Response{i} Answer"] = extract_final_answer(data[key])
        else:
            row[f"Response{i} Answer"] = None

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv('PromptOuput_Reasoner_V1.csv')