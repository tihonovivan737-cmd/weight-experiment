from pathlib import Path

LB, RB, LP, RP = chr(91), chr(93), chr(40), chr(41)
p = Path('/home/mpiscil/weight-experiment/scripts/probe_wave7_report.py')
lines = p.read_text(encoding='utf-8').split('\n')
want = ' ' * 8 + 'a, b = ref' + LB + 'str' + LP + 'l' + RP + RB + ', B' + LB + 'str' + LP + 'l' + RP + RB
fixed = []
for i, ln in enumerate(lines):
    if 'B' + LB + 'str' + LP + 'l' in ln and ln != want:
        indent = len(ln) - len(ln.lstrip(' '))
        lines[i] = ' ' * indent + 'a, b = ref' + LB + 'str' + LP + 'l' + RP + RB + ', B' + LB + 'str' + LP + 'l' + RP + RB
        fixed.append(i + 1)
p.write_text('\n'.join(lines), encoding='utf-8')
print('fixed lines:', fixed)
import ast
ast.parse(p.read_text(encoding='utf-8'))
print('parses ok')
