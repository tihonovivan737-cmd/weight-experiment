import json
import glob

ROOT = '/home/mpiscil/weight-experiment'
A = json.load(open(f'{ROOT}/runs/analysis/probe_w8_model.json'))['layers']
BASE = json.load(open(f'{ROOT}/runs/analysis/probe_base_qwen3.5-0.8b.json'))['layers']
LIN = [l for l in range(24) if l not in (3, 7, 11, 15, 19, 23)]
FULLS = [3, 7, 11, 15, 19, 23]


def load(n):
    f = glob.glob(f'{ROOT}/runs/analysis/probe6_{n}*.json')
    return json.load(open(f[0]))['layers'] if f else None


def head(name, ref, tag):
    B = load(name)
    if B is None:
        print('нет зонда', name)
        return
    print('=====', name, 'vs', tag)
    zs, dz = [], []
    ddec = dbeta = dsp = 0.0
    for l in LIN:
        a, b = ref[str(l)], B[str(l)]
        pct = 100.0 * (b['mean_z_gate'] / a['mean_z_gate'] - 1.0)
        zs.append(pct)
        dz.append((l, pct))
        ddec = max(ddec, abs(b['mean_abs_decay'] - a['mean_abs_decay']))
        dbeta = max(dbeta, max(abs(x - y) for x, y in zip(b['beta_head'], a['beta_head'])))
        dsp = max(dsp, max(abs(x - y) for x, y in zip(b['softplus_a_dt_head'], a['softplus_a_dt_head'])))
    moved = sum(1 for x in zs if abs(x) >= 0.05)
    print(f'  mean_z_gate: сдвинуто {moved}/18 слоёв, крайние значения {min(zs):+.1f}% .. {max(zs):+.1f}%')
    print('  топ-слоёв: ' + ', '.join(f'L{l} {p:+.1f}%' for l, p in sorted(dz, key=lambda t: -abs(t[1]))[:6]))
    print(f'  mean|exp(A_log)| max|d| {ddec:.8f} ; max|d| head beta {dbeta:.4f} ; max|d| softplus(a+dt) {dsp:.4f}')
    for l in FULLS:
        a, b = ref[str(l)], B[str(l)]
        print(f'  L{l}: gate_logit {a["mean_gate_logit"]:+.3f} -> {b["mean_gate_logit"]:+.3f} '
              f'(d {b["mean_gate_logit"] - a["mean_gate_logit"]:+.3f}) ; '
              f'sigmoid {a["mean_gate_sigmoid"]:.3f} -> {b["mean_gate_sigmoid"]:.3f} '
              f'(x{b["mean_gate_sigmoid"] / a["mean_gate_sigmoid"]:.2f}) ; '
              f'max head sigmoid {max(a["gate_sigmoid_head"]):.3f} -> {max(b["gate_sigmoid_head"]):.3f} ; '
              f'rms_k {100 * (b["rms_k"] / a["rms_k"] - 1):+.1f}% ; '
              f'rms_attn_branch {100 * (b["rms_attn_branch"] / a["rms_attn_branch"] - 1):+.1f}% ; '
              f'gain_attn {100 * (b["gain_attn"] / a["gain_attn"] - 1):+.1f}%')


for n in ['w31_z_dominant_restore106', 'w32_qknorm104_on_w8',
          'w33_fullgate_open_closed_heads', 'w34_restore_plus_qknorm104']:
    head(n, A, 'w8')
print()
head('w30_qknorm_sharp108', BASE, 'исходная модель')
head('w24_fullgate_open085', BASE, 'исходная модель')
head('w29_z_total087', A, 'w8')
