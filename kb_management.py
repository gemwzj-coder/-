"""知识库生命周期管理：哈希追踪、自动重建、版本快照、审计日志、数据质量校验。

纯标准库实现，被 rag_graph业绩.py 的 initialize() 调用，也可通过 manage_kb.py 命令行操作。
"""

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
import shutil

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(WORKSPACE, 'kb_state.json')
AUDIT_FILE = os.path.join(WORKSPACE, 'kb_audit.jsonl')
VERSION_DIR = os.path.join(WORKSPACE, 'kb_versions')

_Q_SUFFIXES = ['的业绩表现怎么样', '的业绩表现如何', '各年度收益是多少', '各期收益分别是多少',
               '的收益情况如何', '的基本情况是什么', '具体情况', '怎么样', '呢']


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_question(q: str) -> str:
    for suffix in _Q_SUFFIXES:
        q = q.split(suffix)[0]
    return q.strip().strip('？?')


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def audit(action: str, **fields) -> None:
    entry = {'ts': now_str(), 'action': action}
    entry.update(fields)
    with open(AUDIT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def data_date_of(text: str):
    """从知识库内容里提取数据截止日期（净值日期/截至日期取最大）。"""
    dates = []
    for m in re.finditer(r'(净值日期为|截至|截止)\s*(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})日?', text):
        try:
            dates.append(datetime(int(m.group(2)), int(m.group(3)), int(m.group(4))))
        except ValueError:
            pass
    if not dates:
        return None
    return max(dates).strftime('%Y-%m-%d')


def should_rebuild(knowledge_path: str, persist_dir: str) -> bool:
    """向量库缺失，或知识库文件哈希与上次构建不一致时，需要重建。"""
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return True
    state = load_state()
    current = file_sha256(knowledge_path)
    return state.get('file_sha256') != current


def snapshot_knowledge(knowledge_path: str) -> str:
    """把当前知识库文件快照到 kb_versions/（按哈希去重），返回版本标识。"""
    os.makedirs(VERSION_DIR, exist_ok=True)
    digest = file_sha256(knowledge_path)
    for name in os.listdir(VERSION_DIR):
        if name.endswith('.txt') and name.startswith('knowledge_'):
            if file_sha256(os.path.join(VERSION_DIR, name)) == digest:
                return name.replace('knowledge_', '').replace('.txt', '')
    version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(knowledge_path, os.path.join(VERSION_DIR, f'knowledge_{version_id}.txt'))
    audit('snapshot', version=version_id, file_sha256=digest, file_size=os.path.getsize(knowledge_path))
    return version_id


def record_build(knowledge_path: str, doc_count: int, note: str = '') -> None:
    """构建成功后记录状态 + 审计。"""
    digest = file_sha256(knowledge_path)
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        content = f.read()
    state = {
        'knowledge_path': os.path.abspath(knowledge_path),
        'file_sha256': digest,
        'file_size': os.path.getsize(knowledge_path),
        'docs': doc_count,
        'data_date': data_date_of(content),
        'built_at': now_str(),
    }
    save_state(state)
    audit('build', note=note, **state)
    return state


def list_versions() -> list:
    if not os.path.exists(VERSION_DIR):
        return []
    versions = []
    for name in sorted(os.listdir(VERSION_DIR), reverse=True):
        if name.startswith('knowledge_') and name.endswith('.txt'):
            p = os.path.join(VERSION_DIR, name)
            versions.append({
                'version': name.replace('knowledge_', '').replace('.txt', ''),
                'file': name,
                'size': os.path.getsize(p),
                'sha256': file_sha256(p),
            })
    return versions


def rollback(version_id: str) -> str:
    """把 kb_versions/ 里的某个版本还原为 knowledge.txt，返回恢复后的路径。"""
    src = os.path.join(VERSION_DIR, f'knowledge_{version_id}.txt')
    if not os.path.exists(src):
        raise FileNotFoundError(f'版本不存在: {version_id}')
    target = os.path.join(WORKSPACE, 'knowledge.txt')
    shutil.copy2(src, target)
    audit('rollback', version=version_id, file_sha256=file_sha256(target))
    return target


# ==================== 数据质量校验 ====================

def _metric_value_raw(text: str, label: str):
    # 先剔除"最大回撤X%"片段，避免被误当成收益
    text = re.sub(r'最大回撤[^，。；%]*?[+-]?\d+(?:\.\d+)?%', '', text)
    m = re.search(re.escape(label) + r'[^，。；%]*?([+-]?\d+(?:\.\d+)?)%', text)
    if not m:
        return None
    try:
        return round(float(m.group(1)), 2)
    except ValueError:
        return None


def validate_knowledge(path: str) -> dict:
    """知识库质量检查：结构、重复问答、同产品数值冲突、必填字段。"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = [b.strip() for b in content.split('\n\nQ: ') if b.strip()]
    issues = []
    stats = {'blocks': len(blocks), 'no_a': 0, 'products': 0}
    qs = []
    products = {}
    missing_basic = 0
    basic_names = set()

    for b in blocks:
        if not b.startswith('Q:'):
            b = 'Q: ' + b
        q0 = b.split('\n', 1)[0].replace('Q:', '', 1).strip()
        if re.match(r'^(数据截止|数据日期|净值日期|截止日期|更新日期)', q0):
            continue  # 元数据标记行（如"数据截止 2026-07-31"），不是问答块
        if '\nA: ' not in b:
            stats['no_a'] += 1
            issues.append({'type': 'no_answer', 'block': b[:60]})
            continue
        q, a = b.split('\nA: ', 1)
        q = q.replace('Q:', '', 1).strip()
        a = a.strip()
        qs.append(q)
        name = clean_question(q)
        if '基本情况' in q and '这份' not in q:
            if '管理规模' not in a or '策略类型' not in a:
                missing_basic += 1
                issues.append({'type': 'basic_info_missing', 'product': name})
            else:
                basic_names.add(name)
        if not name:
            continue
        prod = products.setdefault(name, {})
        for metric, label in [('成立以来', '成立以来'),
                              ('今年以来', '今年以来'),
                              ('近1周', '近1周'),
                              ('2025年', '2025年'), ('2024年', '2024年'), ('2023年', '2023年'),
                              ('2022年', '2022年')]:
            v = _metric_value_raw(a, label)
            if v is not None:
                prod.setdefault(metric, set()).add(v)

    stats['products'] = len(products)

    dup_qs = [q for q, c in Counter(qs).items() if c > 1]
    if dup_qs:
        issues.append({'type': 'duplicate_question', 'count': len(dup_qs), 'items': dup_qs[:10]})

    conflicts = []
    for name, metrics in products.items():
        for metric, vals in metrics.items():
            if len(vals) > 1:
                conflicts.append({'product': name, 'metric': metric, 'values': sorted(vals)})
    if conflicts:
        issues.append({'type': 'value_conflict', 'count': len(conflicts), 'items': conflicts[:10]})

    # 业绩文档必填字段：真产品（有基本情况文档）必须同时有 近1周 / 今年以来 / 成立以来
    missing_metrics = []
    product_marker = re.compile(r'(号|期|增强|量化|优选|精选|多策略|指增)')
    for name in basic_names:
        if not product_marker.search(name):
            continue  # 指数/主题文档，不是产品
        metrics = products.get(name, {})
        for m in ['近1周', '今年以来', '成立以来']:
            if m not in metrics:
                missing_metrics.append({'product': name, 'missing': m})
    if missing_metrics:
        issues.append({'type': 'missing_metric', 'count': len(missing_metrics), 'items': missing_metrics[:10]})

    if missing_basic:
        issues.append({'type': 'basic_info_missing_count', 'count': missing_basic})

    return {'path': path, 'stats': stats, 'issues': issues}


def get_kb_info(path: str) -> dict:
    state = load_state()
    info = {
        'path': os.path.abspath(path),
        'current_sha256': file_sha256(path),
        'current_size': os.path.getsize(path),
        'data_date': data_date_of(open(path, encoding='utf-8').read()),
        'state': state if state else None,
        'needs_rebuild': state.get('file_sha256') != file_sha256(path) if state else True,
        'versions': list_versions(),
    }
    return info
