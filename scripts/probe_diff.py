"""Сравнение двух activation-зондов: что реально изменилось в активациях."""
import json, sys
from pathlib import Path

A = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
B = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
na, nb = Path(A["model_dir"]).name, Path(B["model_dir"]).name
la, lb = A["layers"], B["layers"]
same_idx = A["search_indices_used"] == B["search_indices_used"]
print(f"{na} (n={A['n_tasks']})  ->  {nb} (n={B['n_tasks']})   одни и те же задачи: {same_idx}")
L = sorted(la, key=int)
LIN = [k for k in L if la[k]["type"] == "linear_attention"]


def d(col, key_a, key_b=None, tail=False):
    ka = key_a + ("_tail" if tail else "")
    kb = (key_b or key_a) + ("_tail" if tail else "")
    x, y = la[col].get(ka), lb[col].get(kb)
    if x is None or y is None:
        return "        -         "
    pct = (y / x - 1) * 100 if abs(x) > 1e-12 else 0.0
    return f"{x:6.3f}->{y:6.3f} {pct:+6.1f}%"


print(f"{'сл':>3} {'тип':>4} {'gainAttn (всё/хвост)':>30} {'gainMLP (всё/хвост)':>30}")
for k in L:
    print(f"{k:>3} {la[k]['type'][:4]:>4} "
          f"{d(k,'gain_attn')} {d(k,'gain_attn',tail=True)}   {d(k,'gain_mlp')} {d(k,'gain_mlp',tail=True)}")
print("\n{'сл':>3} meanZ(z-вентиль линейных слоёв)   meanBeta   mean|A|decay   softplus(a+dt) по головам (min..max)")
for k in LIN:
    sp = lb[k]["softplus_a_dt_head"]
    print(f"{k:>3} {d(k,'mean_z_gate')} {d(k,'mean_z_gate',tail=True)}  {d(k,'mean_beta')}  "
          f"{d(k,'mean_abs_decay')}  {min(sp):.2f}..{max(sp):.2f}")
print("\nполные слои: gate logit -> sigmoid")
for k in L:
    if la[k]["type"] != "linear_attention":
        print(f"{k:>3} {d(k,'mean_gate_logit')}  {d(k,'mean_gate_sigmoid')}   rms_v {la[k]['rms_v']:.3f}->{lb[k]['rms_v']:.3f}")
