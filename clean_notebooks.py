# Run this once: save as clean_notebooks.py in project root
# then run: python clean_notebooks.py

import json, glob, re

for path in glob.glob('notebooks/*.ipynb'):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        # Clear all output (hides any printed keys)
        if cell.get('outputs'):
            cell['outputs'] = []
            cell['execution_count'] = None
            changed = True
        # Check source for hardcoded keys
        src = ''.join(cell.get('source', []))
        if re.search(r'(gsk_|AIzaSy)[A-Za-z0-9_-]{20,}', src):
            print(f"⚠️  WARNING: Possible API key found in {path}")
            print(f"   Cell preview: {src[:100]}")

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Cleared outputs: {path}")

print("✓ Done. Review any warnings above manually.")