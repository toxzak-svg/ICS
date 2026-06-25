"""Find all tok references in runtime pipeline.py."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
with open("/content/ICS/ics/pipeline.py") as f:
    content = f.read()
# Find all lines that reference tok as a name (not as part of "tokenize", "token", etc.)
for i, line in enumerate(content.splitlines(), 1):
    # crude check: word boundary "tok" not part of longer word
    import re
    for m in re.finditer(r"\btok\b", line):
        # Check if it's part of "tokenize", "tokenizer", etc.
        idx = m.start()
        before = line[max(0,idx-2):idx]
        after = line[idx:idx+10]
        if "tokenize" in after or "tokenizer" in after or "token_id" in after or "tok_" in after:
            continue
        print(f"  line {i} col {idx+1}: {line.rstrip()}")
        break
'''
r = c.exec(script)
print(r.stdout)
