#!/usr/bin/env python3
"""
统计 info 文件中 #ifdef KRL 块内的行覆盖率。
用法: python3 krl_line_cov.py <coverage.info> <source_root>
"""
import re, sys, os

info_file = sys.argv[1]
src_root = sys.argv[2].rstrip('/')

# ---- 1. 从 coverage.info 提取每个源文件的命中数据 ----
file_hits = {}   # file_path -> {line_num: hit_count}
current_sf = None
with open(info_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith('SF:'):
            current_sf = line[3:]
            file_hits.setdefault(current_sf, {})
        elif line.startswith('DA:'):
            m = re.match(r'DA:(\d+),(\d+)', line)
            if m and current_sf:
                file_hits[current_sf][int(m.group(1))] = int(m.group(2))
        elif line == 'end_of_record':
            current_sf = None

# ---- 2. 收集需要统计的源文件 ----
candidate_dirs = ['faiss', 'faiss/impl', 'faiss/impl/fast_scan', 'faiss/utils']
source_files = []
for d in candidate_dirs:
    dd = os.path.join(src_root, d)
    if os.path.isdir(dd):
        for fn in os.listdir(dd):
            if fn.endswith(('.cpp', '.h')):
                source_files.append(os.path.join(d, fn))

# ---- 3. 对每个源文件提取 KRL 行范围 ----
def extract_krl_ranges(filepath):
    """提取 #ifdef KRL / #ifndef KRL 块的行号范围。
    返回 list of (start, end) 1-based 闭区间。"""
    ranges = []
    stack = []      # (directive_line, is_krl_block)
    with open(filepath) as f:
        lines = f.readlines()

    # 预处理: 处理行尾注释, 去除续行符
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('#ifdef KRL') or stripped.startswith('#if defined(KRL)'):
            stack.append((i+1, True))
        elif stripped.startswith('#ifndef KRL'):
            stack.append((i+1, False))
        elif stripped.startswith('#if '):
            # 看是否包含 KRL
            if 'KRL' in stripped:
                stack.append((i+1, True))
            else:
                stack.append((i+1, False))
        elif stripped.startswith('#elif'):
            # 如果是 KRL 相关的 #elif，可能需要处理
            if stack and stack[-1][1]:
                # 记录上一个块
                pass
        elif stripped.startswith('#else'):
            if stack and stack[-1][1]:
                ranges.append((stack[-1][0], i))
                stack[-1] = (i+1, False)
        elif stripped.startswith('#endif'):
            if stack:
                start, is_krl = stack.pop()
                if is_krl:
                    ranges.append((start, i+1))
        i += 1

    # 合并重叠/相邻区间
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for r in ranges[1:]:
        if r[0] <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged

# ---- 4. 匹配 info 文件中的路径 ----
def match_info_path(src_rel, file_hits):
    """在 file_hits 的 key 中找到匹配 src_rel 的路径。"""
    for k in file_hits:
        if k.endswith('/' + src_rel) or k.endswith('\\' + src_rel):
            return k
    # 模糊匹配
    basename = os.path.basename(src_rel)
    for k in file_hits:
        if k.endswith('/' + basename) or k.endswith('\\' + basename):
            # 进一步匹配更多路径片段
            parts = src_rel.replace('\\', '/').split('/')
            if all(p in k for p in parts[-3:]):
                return k
    return None

# ---- 5. 计算覆盖率 ----
total_krl_lines = 0
covered_krl_lines = 0
per_file = []

for src_rel in sorted(source_files):
    src_path = os.path.join(src_root, src_rel)
    if not os.path.isfile(src_path):
        continue
    krl_ranges = extract_krl_ranges(src_path)
    if not krl_ranges:
        continue

    info_key = match_info_path(src_rel, file_hits)
    if not info_key:
        continue
    hits = file_hits[info_key]

    file_total = 0
    file_covered = 0
    for start, end in krl_ranges:
        for ln in range(start, end + 1):
            if ln in hits:
                file_total += 1
                if hits[ln] > 0:
                    file_covered += 1

    if file_total > 0:
        pct = 100.0 * file_covered / file_total
        per_file.append((src_rel, file_covered, file_total, pct))
        total_krl_lines += file_total
        covered_krl_lines += file_covered

# ---- 6. 输出 ----
print("=" * 70)
print(f"{'File':<45} {'Covered':>8} {'Total':>8} {'Rate':>7}")
print("-" * 70)
for name, cov, tot, pct in per_file:
    flag = '✅' if pct >= 90 else ('⚠️' if pct > 0 else '❌')
    print(f"{flag} {name:<42} {cov:>8} {tot:>8} {pct:>6.1f}%")
print("-" * 70)
overall = 100.0 * covered_krl_lines / total_krl_lines if total_krl_lines > 0 else 0
print(f"{'TOTAL (KRL blocks only)':<45} {covered_krl_lines:>8} {total_krl_lines:>8} {overall:>6.1f}%")
print("=" * 70)
print(f"\n  KRL 块总行数:     {total_krl_lines}")
print(f"  已覆盖行数:       {covered_krl_lines}")
print(f"  KRL 块覆盖率:     {overall:.1f}%")
print()
