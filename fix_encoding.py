# -*- coding: utf-8 -*-
import re

with open(r'e:\fbi ML\data_audit.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace non-ASCII with ASCII equivalents
replacements = {
    u'\u2014': '--',
    u'\u2013': '-',
    u'\u2192': '->',
    u'\u2194': '<->',
    u'\u2264': '<=',
    u'\u2265': '>=',
    u'\u00d7': 'x',
    u'\u2026': '...',
    u'\u2018': "'",
    u'\u2019': "'",
    u'\u201c': '"',
    u'\u201d': '"',
    u'\u2022': '*',
    u'\u2713': 'OK',
    u'\u2717': 'X',
    u'\u2190': '<-',
    u'\u21d4': '<=>',
    u'\u21d2': '=>',
    u'\u2191': '^',
    u'\u2193': 'v',
    u'\u2764': '<3',
    u'\u00b2': '2',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(r'e:\fbi ML\data_audit.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done cleaning special characters')
