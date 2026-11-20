#!/usr/bin/env python3
"""Re-evaluate report variants on GSM8K test and the previously used holdout.

Run from the server checkout: .venv/bin/python scripts/run_report_validation.py --help
The evaluation deliberately uses eval_gsm8k.py unchanged, batch=1 and greedy decoding.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'models/qwen3.5-0.8b'
VARIANTS = ROOT / 'runs/variants'
STAGING = ROOT / 'runs/report_validation/models'
OUT = ROOT / 'runs/report_validation/results'
SPLIT = ROOT / 'runs/splits/holdout6.json'
REUSE = {
    'orig': {'test': 'runs/baseline_test', 'holdout6': 'runs/holdout6/orig'},
    'w8': {'test': 'runs/variants/w8_z_gate_mild_format/test', 'holdout6': 'runs/holdout6/w8'},
    'w50': {'test': 'runs/variants/w50_finalrow_rotation15_smallmargin/test', 'holdout6': 'runs/holdout6/w50_finalrow_rotation15_smallmargin'},
}
NEW_VARIANTS = ('w36', 'w30', 'w83', 'w87', 'w12')
PROMPT_SUFFIX = '\n\nSolve the problem step by step. Finish with a final line exactly in this format: FINAL: <number>'
MODELS = {
    'orig': ('', '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696', None),
    'w8': ('w8_z_gate_mild_format', '296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55', 'orig'),
    'w50': ('w50_finalrow_rotation15_smallmargin', '544d4eea6ab1d1532e28c134366f29720f084c94281b7b04d0c2a4c219881e70', 'w8'),
    'w36': ('w36_z_late_extra094', '943a08f0e95cb019e361ef805461faaf15f5321f4d7d94fff676d69b99650335', 'w8'),
    'w30': ('w30_qknorm_sharp108', '8ff9d188751e77068f476558d84667dfcd0793ca953df8128f1160d34716ee10', 'w8'),
    'w83': ('w83_mlp_layer_lever_on_w8', 'bdf25aa4445cf359f30320b9584532fe0265c5f93f72c42288c099c3bdc7f72e', 'w8'),
    'w87': ('w87_stack_finalrow_on_w83', '6dbd6b150a63cfce4a0777d1cfe56b538eb92663581bf4d81377c9a13aa954b7', 'w83'),
    'w12': ('w12_z_gate_strong', '486902198597bbed611565ec1c93f2dd2822ccb0f5d4106f8af2e1f1ba191471', 'orig'),
}

def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()

def weights(directory):
    return sorted(p for p in directory.glob('*.safetensors') if p.is_file())

def check_model(directory, expected):
    if not (directory / 'config.json').is_file():
        return False
    files = weights(directory)
    if len(files) != 1:
        return False
    actual = sha(files[0])
    if actual != expected:
        raise RuntimeError(f'Неверный SHA-256 {files[0]}: {actual}; ожидался {expected}')
    return True

def resources(gpus, min_ram, min_gpu):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, *v = line.split()
        if key == 'MemAvailable:': mem['available'] = int(v[0]) // 1024
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {int(a.strip()): int(b.strip()) for a, b in (line.split(',') for line in output.splitlines())}
    print(f'RAM доступно {mem.get("available", 0)} MiB; GPU свободно: {free}', flush=True)
    if mem.get('available', 0) < min_ram:
        raise RuntimeError(f'Недостаточно RAM: нужно не менее {min_ram} MiB')
    for gpu in gpus:
        if free.get(gpu, 0) < min_gpu:
            raise RuntimeError(f'GPU {gpu}: нужно не менее {min_gpu} MiB свободной VRAM')

def prepare_models(build):
    resolved = {}
    for label, (name, expected, parent) in MODELS.items():
        if label == 'orig':
            if not check_model(BASE, expected):
                raise RuntimeError(f'Отсутствует исходная модель или файл весов: {BASE}')
            resolved[label] = BASE
            continue
        existing = VARIANTS / name / 'model'
        staged = STAGING / name / 'model'
        if check_model(existing, expected):
            resolved[label] = existing
        elif check_model(staged, expected):
            resolved[label] = staged
        elif not build:
            raise RuntimeError(f'Нет весов {label}: {existing}. Запустите с --build-missing для восстановления.')
        else:
            target = STAGING / name
            recipe = ROOT / 'recipes' / (name + '.json')
            if not recipe.is_file():
                raise RuntimeError(f'Нет рецепта: {recipe}')
            if target.exists():
                raise RuntimeError(f'Неполная или некорректная сборка {target}; проверьте её вручную.')
            if shutil.disk_usage(ROOT).free < 3 * 1024**3:
                raise RuntimeError('Для сборки требуется не менее 3 GiB свободного места')
            print(f'Восстанавливаю {label} в {target} из {parent}...', flush=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/modify_weights.py'), '--recipe', str(recipe), '--base', str(resolved[parent]), '--out', str(target)], cwd=ROOT, check=True)
            if not check_model(staged, expected):
                raise RuntimeError(f'Не найдены собранные веса: {staged}')
            resolved[label] = staged
        print(f'{label}: SHA-256 OK, {resolved[label]}', flush=True)
    return resolved

def run_job(label, split, gpu, model, spec, run_dir, threads):
    destination = run_dir / label / split
    destination.mkdir(parents=True, exist_ok=True)
    result_file = destination / 'results.jsonl'
    if result_file.exists():
        raw = result_file.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise RuntimeError(f'Неполная последняя строка {result_file}; сохраните файл и восстановите его вручную')
        seen = set()
        for line in raw.splitlines():
            record = json.loads(line)
            if record['index'] in seen:
                raise RuntimeError(f'Повтор индекса {record["index"]} в {result_file}')
            seen.add(record['index'])
    cmd = [sys.executable, str(ROOT/'scripts/eval_gsm8k.py'), '--model', str(model), '--device', 'cuda:0', '--dtype', 'bfloat16', '--batch-size', '1', '--max-new-tokens', '1024', '--threads', str(threads), '--out', str(destination), '--note', f'report validation {label} {split}']
    if split == 'test': cmd += ['--split', 'test']
    else: cmd += ['--split', 'custom', '--indices', spec]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['OMP_NUM_THREADS'] = str(threads)
    print(f'GPU {gpu}: {label} / {split} -> {destination}', flush=True)
    with (destination/'eval.log').open('a', encoding='utf8') as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f'{label}/{split}, GPU {gpu}: код {rc}, см. {destination / "eval.log"}')
    data = json.loads((destination/'summary.json').read_text(encoding='utf8'))['summary']
    expected = 1319 if split == 'test' else 400
    if data['completed'] != expected or data['total'] != expected:
        raise RuntimeError(f'{label}/{split}: неполный прогон {data}')
    print(f'GPU {gpu}: {label}/{split} готово: {data["correct"]}/{expected}', flush=True)

def reuse_results(run_dir):
    holdout = json.loads(SPLIT.read_text(encoding='utf8'))
    for label, splits in REUSE.items():
        for split, relative in splits.items():
            source = ROOT / relative
            results = source / 'results.jsonl'
            summary = json.loads((source / 'summary.json').read_text(encoding='utf8'))
            conf = summary['config']
            expected_ids = list(range(1319)) if split == 'test' else sorted(holdout['holdout_indices'])
            expected_hash = hashlib.sha256(','.join(map(str, expected_ids)).encode()).hexdigest()
            weight_info = conf.get('model_info', {}).get('weights', {})
            weight_hashes = [value.get('sha256') for value in weight_info.values()]
            if (weight_hashes != [MODELS[label][1]] or conf.get('indices_sha256') != expected_hash
                or conf.get('batch_size') != 1 or conf.get('max_new_tokens') != 1024
                or conf.get('dtype') != 'bfloat16' or conf.get('do_sample') is not False
                or conf.get('prompt_suffix') != PROMPT_SUFFIX):
                raise RuntimeError(f'Настройки или хеши сохранённого прогона не совпали: {source}')
            data = [json.loads(line) for line in results.read_text(encoding='utf8').splitlines() if line.strip()]
            actual_ids = [row['index'] for row in data]
            if (len(data) != len(expected_ids) or len(set(actual_ids)) != len(data)
                or sorted(actual_ids) != expected_ids
                or sum(row['correct'] for row in data) != summary['summary']['correct']):
                raise RuntimeError(f'Сохранённый прогон неполный или повреждён: {results}')
            dest = run_dir / label / split
            dest.mkdir(parents=True, exist_ok=True)
            for name in ('results.jsonl', 'summary.json'):
                src = source / name
                target = dest / name
                if target.exists():
                    if sha(target) != sha(src):
                        raise RuntimeError(f'Не перезаписываю существующий файл: {target}. Выберите новый --run-id.')
                else:
                    shutil.copy2(src, target)
            (dest / 'provenance.json').write_text(json.dumps({'source': str(source), 'results_sha256': sha(results), 'reused': True}, indent=2), encoding='utf8')
            print(f'Используется сохранённый прогон: {label}/{split}, {len(data)} задач', flush=True)

def write_comparison(run_dir):
    rows = []
    for split in ('test', 'holdout6'):
        reference = {}
        for label in MODELS:
            dest = run_dir / label / split
            data = [json.loads(line) for line in (dest/'results.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
            by_index = {row['index']: row for row in data}
            expected = 1319 if split == 'test' else 400
            if len(data) != expected or len(by_index) != expected:
                raise RuntimeError(f'{dest}: неполный файл или повтор индексов')
            if label == 'orig': reference = by_index
            if by_index.keys() != reference.keys():
                raise RuntimeError(f'{dest}: набор индексов отличается от исходной модели')
            if any(by_index[i]['question'] != reference[i]['question'] or by_index[i]['expected'] != reference[i]['expected'] for i in reference):
                raise RuntimeError(f'{dest}: вопросы или эталонные ответы не совпадают с исходной моделью')
            correct = sum(row['correct'] for row in data)
            fixed = sum(not reference[i]['correct'] and by_index[i]['correct'] for i in reference)
            broken = sum(reference[i]['correct'] and not by_index[i]['correct'] for i in reference)
            rows.append([split, label, expected, correct, round(100*correct/expected, 2), sum(row['answer_extracted'] for row in data), sum(row['truncated'] for row in data), fixed, broken])
    with (run_dir/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'model', 'tasks', 'correct', 'accuracy_percent', 'answers_extracted', 'truncated', 'fixed_vs_orig', 'broken_vs_orig'])
        w.writerows(rows)
    print(f'Сравнение: {run_dir / "comparison.csv"}', flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-missing', action='store_true', help='восстановить недостающие варианты в отдельной папке, сверить SHA-256')
    ap.add_argument('--prepare-only', action='store_true', help='проверить или восстановить модели без оценки')
    ap.add_argument('--run-id', default='report-v2', help='отдельный каталог результатов; повторный запуск продолжает прерванный')
    ap.add_argument('--gpus', default='0,1', help='физические индексы GPU, по одному процессу на GPU')
    ap.add_argument('--threads', type=int, default=8, help='CPU потоки на процесс')
    ap.add_argument('--min-ram-mib', type=int, default=10950)
    ap.add_argument('--min-gpu-mib', type=int, default=6943)
    args = ap.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '-_' for c in args.run_id): ap.error('run-id: только буквы, цифры, дефис и подчёркивание')
    if args.threads < 1: ap.error('threads должен быть положительным')
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0: ap.error('укажите уникальные индексы GPU')
    if ROOT != Path('/home/mpiscil/weight-experiment'):
        raise RuntimeError('eval_gsm8k.py использует /home/mpiscil/weight-experiment; запустите в исходном расположении')
    for split in ('test', 'train'):
        path = ROOT/'datasets/gsm8k/main'/f'{split}-00000-of-00001.parquet'
        if not path.is_file(): raise RuntimeError(f'Нет датасета: {path}')
    spec = subprocess.check_output([sys.executable, str(ROOT/'scripts/holdout_indices.py')], cwd=ROOT, text=True).strip()
    resources(gpus, args.min_ram_mib, args.min_gpu_mib)
    run_dir = OUT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (OUT/'.report_validation.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Этот run-id уже запущен: {args.run_id}')
        models = prepare_models(args.build_missing)
        if args.prepare_only: return
        reuse_results(run_dir)
        print('ВНИМАНИЕ: повторная оценка всех вариантов раскрывает test и holdout; далее они не являются независимыми для выбора.', flush=True)
        jobs = queue.Queue()
        stop = threading.Event()
        for split in ('test', 'holdout6'):
            for label in NEW_VARIANTS: jobs.put((label, split))
        def worker(gpu):
            while not stop.is_set():
                try: label, split = jobs.get_nowait()
                except queue.Empty: return
                try:
                    run_job(label, split, gpu, models[label], spec, run_dir, args.threads)
                except Exception:
                    stop.set()
                    raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for f in futures: f.result()
        write_comparison(run_dir)
        print(f'Готово: {run_dir}', flush=True)

if __name__ == '__main__':
    try: main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f'ОШИБКА: {exc}', file=sys.stderr)
        sys.exit(1)
