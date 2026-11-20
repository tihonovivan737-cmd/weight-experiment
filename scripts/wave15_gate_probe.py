#!/usr/bin/env python3
"""Замер, какие строки mlp.gate_proj двигают финальный логит в пользу ответа и в пользу
FINAL (волна 15, подготовительный зонд, единственный GPU-прогон, только чтение w8).

Ничего не правится и ничего не обучается: скрипт только измеряет. База — w8
(sha256 296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55), данные —
только разрешённые: те же 63 задачи search, что в `probe_w8_model.json`, и тексты ответов
w8 из `runs/variants/w8_z_gate_mild_format/search/results.jsonl`.

Смысл. Финальный логит токена t = (n_final ⊙ h_final) · E_t. Вклад строки j MLP-блока слоя L
в h_final равен a_j · (D_j · (n_final ⊙ E_t)), где a_j = gelu(g_j) · u_j, D_j — строка j
down_proj. Значит вклад измерим без градиентов:

    C_ans_j = среднее по позициям, где текст w8 пишет итоговый ответ,
              a_j · (D_j · (n_final ⊙ E_gold_j_task))
    C_fin_j = среднее по позициям, где текст w8 пишет токен FINAL,
              a_j · (D_j · (n_final ⊙ E_FINAL))
    d_j     = C_ans_j − C_fin_j

Строкой в logit-пространстве gate-строка не является: её связь с читалкой идёт через D_j,
поэтому замер отвечает на вопрос «какие gate-строки уже работают в пользу ответа, а какие
в пользу FINAL». Это basis для детерминированного масштабирования построчно.

Запуск (один GPU-прогон, физический GPU1 -> cuda:0):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/wave15_gate_probe.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import struct
import re as _re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")
W8 = ROOT / "runs/variants/w8_z_gate_mild_format"
FULL = [3, 7, 11, 15, 19, 23]
FINAL_ROWS = {95429, 64697}
OUT_JSON = ROOT / "runs/analysis/wave15_gate_probe.json"
OUT_TXT = ROOT / "runs/analysis/wave15_gate_probe.txt"
DTYPE_MAP = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def read_header(fp):
    """Канонический заголовок safetensors: 8 байт длины, затем JSON (utf-8)."""
    n = struct.unpack("<Q", fp.read(8))[0]
    return 8 + n, json.loads(fp.read(n).decode("utf-8"))


def get_tensor(fp, hs, h, name):
    """Прочитать один тензор целиком и привести к форме из метаданных."""
    info = h[name]
    fp.seek(hs + info["data_offsets"][0])
    raw = fp.read(info["data_offsets"][1] - info["data_offsets"][0])
    t = torch.frombuffer(bytearray(raw), dtype=DTYPE_MAP[info["dtype"]])
    return t.reshape(info["shape"]).float()


def load_probe_module():
    spec = importlib.util.spec_from_file_location("flp", ROOT / "scripts" / "final_logit_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    dev = "cuda:0"
    flp = load_probe_module()
    eval_mod = flp.load_eval_module()
    items = flp.build_tasks(63)
    if len(items) < 40:
        raise SystemExit(f"задач для замеров: {len(items)} — недостаточно")

    # правильность по w8 берём из уже посчитанных результатов search (поля index/correct)
    ok_map = {}
    for line in (W8 / "search/results.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        ok_map[int(rec["index"])] = bool(rec.get("correct"))

    model_dir = W8 / "model"
    shards = sorted(model_dir.rglob("*.safetensors"))
    if not shards:
        raise SystemExit("шард w8 не найден")
    shard = shards[0]
    with open(shard, "rb") as fp:
        hs, h = read_header(fp)
        E = get_tensor(fp, hs, h, "model.language_model.embed_tokens.weight").to(dev)
        n_final = get_tensor(fp, hs, h, "model.language_model.norm.weight").to(dev)
        # down_proj.weight имеет форму [1024, 3584]: строка j gate соответствует СТОЛБЦУ j,
        # поэтому B = транспонированный down_proj, масштабированный нормой читалки
        B = {L: get_tensor(fp, hs, h, f"model.language_model.layers.{L}.mlp.down_proj.weight").T
             .to(dev) * n_final for L in FULL}

    row_final = E[95429]
    gain_final = {L: B[L] @ row_final for L in FULL}

    device = dev
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True).to(device)
    model.eval()

    layers = {}
    for name, mod in model.named_modules():
        m = re.fullmatch(r"model\.language_model\.layers\.(\d+)", name)
        if m:
            layers[int(m.group(1))] = mod

    cur: dict[str, object] = {}
    acc = {L: {"C_ans_ok": torch.zeros(3584, device=device),
               "C_ans_bad": torch.zeros(3584, device=device),
               "C_gold_ok": torch.zeros(3584, device=device),
               "C_gold_bad": torch.zeros(3584, device=device),
               "C_fin": torch.zeros(3584, device=device),
               "act_sum": torch.zeros(3584, device=device),
               "gain_gold_sum": torch.zeros(3584, device=device),
               "n_active": torch.zeros(3584, device=device),
               "C_fin": torch.zeros(3584, device=device),
               "n_ans_ok": 0, "n_ans_bad": 0, "n_fin": 0} for L in FULL}

    # декомпозиция P2: вклад строки = (средняя активация) x (среднее усиление),
    # плюс доля позиций, где gelu(g) не нулевая
    marg = {"ok": [], "bad": []}

    # золотой ответ по каждой из 63 задач: строка embed_tokens для последнего токена эталона
    df = pd.read_parquet(flp.DATASET / "train-00000-of-00001.parquet")
    gold_row = {}
    gold_tok = {}
    for i in range(len(df)):
        g = str(df.iloc[i]["answer"])
        m = _re.search(r"####\s*(-?\d+)", g)
        if not m:
            continue
        toks = processor.tokenizer.encode(m.group(1))
        if toks:
            gold_row[i] = E[int(toks[-1])]
            gold_tok[i] = int(toks[-1])

    def hook_gate(L):
        def fn(m, inp, out):
            p = cur.get("pos_ans")
            if p:
                cur.setdefault("g", {})[L] = out[0][torch.tensor(p, device=device)].float()
            pf = cur.get("pos_fin")
            if pf:
                cur.setdefault("gf", {})[L] = out[0][torch.tensor(pf, device=device)].float()
        return fn

    def hook_up(L):
        def fn(m, inp, out):
            p = cur.get("pos_ans")
            if p:
                cur.setdefault("u", {})[L] = out[0][torch.tensor(p, device=device)].float()
            pf = cur.get("pos_fin")
            if pf:
                cur.setdefault("uf", {})[L] = out[0][torch.tensor(pf, device=device)].float()
        return fn

    for L in FULL:
        layers[L].mlp.gate_proj.register_forward_hook(hook_gate(L))
        layers[L].mlp.up_proj.register_forward_hook(hook_up(L))

    n_tasks_used = 0
    n_ok = 0
    with torch.no_grad():
        for idx, question, answer in items:
            gold = str(answer).strip()
            inputs = eval_mod.build_inputs(processor, question)
            ids = inputs["input_ids"][0].to(device)
            ans_ids = processor.tokenizer.encode(gold, add_special_tokens=False)[:420]
            if not ans_ids:
                continue
            seq = torch.cat([ids, torch.tensor(ans_ids, device=device, dtype=ids.dtype)], dim=-1)
            attn = torch.ones_like(seq).unsqueeze(0)
            # маски позиций считаются ДО прохода: позиция p предсказывает tgt[p]
            tgt = seq[1:]
            n = int(tgt.shape[0])
            pos_fin = [p for p in range(n) if int(tgt[p]) in FINAL_ROWS]
            pos_ans = [n - 1]
            is_ok = bool(ok_map.get(int(idx), False))
            cur.clear()
            cur["pos_ans"] = pos_ans
            cur["pos_fin"] = pos_fin
            cur["target_row_ok"] = is_ok
            out_logits = model(input_ids=seq.unsqueeze(0), attention_mask=attn).logits
            lg = out_logits[0, pos_ans[0]]
            am = int(torch.argmax(lg))
            marg["ok" if is_ok else "bad"].append((int(idx), int(am), float(lg[am] - lg[gold_tok[int(idx)]])))
            # вклад каждой gate-строки по выбранным позициям
            for L in FULL:
                g = cur.get("g", {}).get(L)
                u = cur.get("u", {}).get(L)
                if g is None or u is None:
                    continue
                a = torch.nn.functional.gelu(g) * u
                # E по строке предсказываемого токена (то, что реально написала модель)
                tok = int(tgt[pos_ans[0]])
                contrib_ans = (a * (B[L] @ E[tok]).unsqueeze(0)).mean(dim=0)
                # E по строке золотого ответа: голос этой строки в пользу верного числа
                gain_gold = B[L] @ gold_row[int(idx)]
                contrib_gold = (a * gain_gold.unsqueeze(0)).mean(dim=0)
                acc[L]["act_sum"] += a.reshape(-1)
                acc[L]["gain_gold_sum"] += gain_gold
                acc[L]["n_active"] += (torch.nn.functional.gelu(g) != 0).sum(dim=0).reshape(-1).float()
                if cur["target_row_ok"]:
                    acc[L]["C_ans_ok"] += contrib_ans
                    acc[L]["C_gold_ok"] += contrib_gold
                    acc[L]["n_ans_ok"] += 1
                else:
                    acc[L]["C_ans_bad"] += contrib_ans
                    acc[L]["C_gold_bad"] += contrib_gold
                    acc[L]["n_ans_bad"] += 1
                gf = cur.get("gf", {}).get(L)
                uf = cur.get("uf", {}).get(L)
                if gf is not None and uf is not None:
                    af = torch.nn.functional.gelu(gf) * uf
                    contrib_fin = (af * gain_final[L].unsqueeze(0)).mean(dim=0)
                    acc[L]["C_fin"] += contrib_fin
                    acc[L]["n_fin"] += 1
            n_tasks_used += 1
            if is_ok:
                n_ok += 1
            del seq, attn, ids, tgt

    rows = []
    for L in FULL:
        a = acc[L]
        ca_ok = acc[L]["C_ans_ok"] / max(a["n_ans_ok"], 1)
        cf = acc[L]["C_fin"] / max(a["n_fin"], 1)
        dt = ca_ok - cf
        d = dt.cpu().tolist()
        cg_ok = acc[L]["C_gold_ok"] / max(a["n_ans_ok"], 1)
        cg_bad = acc[L]["C_gold_bad"] / max(a["n_ans_bad"], 1)
        gm = (acc[L]["C_gold_ok"] + acc[L]["C_gold_bad"]) / max(a["n_ans_ok"] + a["n_ans_bad"], 1)
        top_d = sorted(range(3584), key=lambda j: -d[j])[:12]
        top_fin = sorted(range(3584), key=lambda j: -cf[j].item())[:16]
        top_gold = sorted(range(3584), key=lambda j: -gm[j].item())[:16]
        # согласованные «голоса за верный ответ»: положительный вклад и на верных, и на неверных
        cons = [j for j in range(3584) if cg_ok[j].item() > 0.0 and cg_bad[j].item() > 0.0]
        top_cons = sorted(cons, key=lambda j: -gm[j].item())[:16]

        n_pos = a["n_ans_ok"] + a["n_ans_bad"]
        base_act = float(acc[L]["act_sum"].abs().sum()) / max(n_pos, 1) / 3584.0
        base_gain = float(acc[L]["gain_gold_sum"].abs().sum()) / max(n_pos, 1) / 3584.0

        cab = acc[L]["C_ans_bad"]
        cflip = cg_bad - cab
        idx_flip = sorted(range(3584), key=lambda j: -cflip[j].item())[:16]
        def pack(idx):
            out = []
            for j in idx:
                ma = float(acc[L]["act_sum"][j]) / max(n_pos, 1)
                mg = float(acc[L]["gain_gold_sum"][j]) / max(n_pos, 1)
                out.append({"row": j,
                            "c_gold_ok": round(cg_ok[j].item(), 4),
                            "c_gold_bad": round(cg_bad[j].item(), 4),
                            "c_gold_mean": round(gm[j].item(), 4),
                            "mean_activation": round(ma, 4),
                            "mean_gain": round(mg, 4),
                            "act_ratio_vs_base": round(ma / max(base_act, 1e-12), 2),
                            "gain_ratio_vs_base": round(mg / max(base_gain, 1e-12), 2),
                            "active_share": round(float(acc[L]["n_active"][j]) / max(n_pos, 1), 3),
                            "c_ans_bad": round(cab[j].item(), 4),
                            "flip_potential": round(cg_bad[j].item() - cab[j].item(), 4),
                            "c_fin": round(cf[j].item(), 4), "d": round(d[j], 4)})
            return out

        rows.append({
            "layer": L,
            "sum_contrib_ans_ok": float(ca_ok.sum()),
            "sum_contrib_fin": float(cf.sum()),
            "sum_d": float(dt.sum()),
            "sum_gold_mean": float(gm.sum()),
            "n_consistent": len(cons),
            "sum_gold_ok": float(cg_ok.sum()),
            "sum_gold_bad": float(cg_bad.sum()),
            "n_ans_ok": a["n_ans_ok"],
            "n_ans_bad": a["n_ans_bad"],
            "n_fin": a["n_fin"],
            "top_rows": pack(top_d),
            "top_by_gold": pack(top_gold),
            "top_by_gold_consistent": pack(top_cons),
            "top_by_final": pack(top_fin),
            "base_mean_activation": round(base_act, 5),
            "base_mean_gain": round(base_gain, 5),
            "margin_ok": round(sum(v[2] for v in marg["ok"]) / max(len(marg["ok"]), 1), 3),
            "margin_bad": round(sum(v[2] for v in marg["bad"]) / max(len(marg["bad"]), 1), 3),
            "margins_bad": [[v[0], v[1], round(v[2], 3)] for v in marg["bad"]],
            "margins_ok": [[v[0], v[1], round(v[2], 3)] for v in marg["ok"]],
            "n_margin_bad_le_1": sum(1 for v in marg["bad"] if v[2] <= 1.0),
            "n_margin_bad_le_3": sum(1 for v in marg["bad"] if v[2] <= 3.0),
            "sum_flip_all": round(float(cflip.sum()), 4),
            "sum_flip_pos": round(float(cflip.clamp(min=0).sum()), 4),
            "n_flip_pos": int((cflip > 0).sum()),
            "top_by_flip": pack(idx_flip),
            "c_gold_rows": [round(float(v), 5) for v in gm.cpu().tolist()],
            "c_fin_rows": [round(float(v), 5) for v in cf.cpu().tolist()],
        })

    lines = [
        "Замер вклада строк mlp.gate_proj в финальный логит (прямой проход, без градиентов).",
        f"База: w8 ({shard.name}). Задач: {n_tasks_used}, из них верных по w8: {n_ok}.",
        "C_ans — средний вклад строки в логит токена ответа; C_fin — вклад в логит FINAL;",
        "d = C_ans − C_fin (положительный = строка работает в пользу ответа).",
        "",
        f"{'слой':>5} {'сумма C_ans(верные)':>19} {'сумма C_fin':>13} {'сумма d':>10} "
        f"{'позиций ответа':>15} {'позиций FINAL':>14}",
    ]
    for r in rows:
        lines.append(f"{r['layer']:>5} {r['sum_contrib_ans_ok']:>19.3f} {r['sum_contrib_fin']:>13.3f} "
                     f"{r['sum_d']:>10.3f} {r['n_ans_ok']:>15} {r['n_fin']:>14}")
    lines.append("")
    for r in rows:
        for key, title in (("top_by_gold", "топ-16 по среднему голосу за ЗОЛОТОЙ ответ"),
                           ("top_by_gold_consistent", f"согласованные голоса за золото (n = {r['n_consistent']})"),
                           ("top_by_final", "топ-16 по C_fin (работают в пользу FINAL)")):
            lines.append(f"слой {r['layer']}: {title}")
            for t in r[key]:
                lines.append(f"  {t['row']:>5} gold_ok {t['c_gold_ok']:>8.4f}  gold_bad {t['c_gold_bad']:>8.4f}  "
                             f"gold_mean {t['c_gold_mean']:>8.4f}  fin {t['c_fin']:>8.4f}  "
                             f"act {t['mean_activation']:>8.4f} (x{t['act_ratio_vs_base']:>5.2f})  "
                             f"gain {t['mean_gain']:>8.4f} (x{t['gain_ratio_vs_base']:>5.2f})  "
                             f"active {t['active_share']:>6.3f}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"n_tasks_used": n_tasks_used, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
