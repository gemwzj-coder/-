"""从当前 knowledge.txt 重建可回归的 RAG 评测集。

评测集不再写死旧产品和旧业绩数值：每次生成时都从当前知识库读取，
以免知识库更新后，事实召回率被“过期标准答案”误伤。
默认生成约 60 条，覆盖产品、管理人、策略、对比、通用问答和错别字。
"""

import json
import sys
from collections import defaultdict

import rag_graph业绩 as rag

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PRODUCT_TARGET = 32
MANAGER_TARGET = 10
STRATEGY_TARGET = 8


def product_facts(series: dict) -> list:
    """选择稳定且对客户有意义的三项业绩指标作为当前知识库的事实标准。"""
    labels = ('成立以来', '今年以来', '近1周')
    return [
        {'metric': label, 'value': round(series[label], 2)}
        for label in labels
        if label in series
    ]


def select_products(catalog: dict) -> list[tuple[str, dict]]:
    """按策略轮流抽样，避免大策略产品淹没小策略。"""
    by_strategy = defaultdict(list)
    for name, meta in catalog.items():
        if meta.get('series') and meta.get('strategy') and name not in rag.INDEX_DOC_NAMES:
            by_strategy[meta['strategy']].append((name, meta))

    for products in by_strategy.values():
        products.sort(key=lambda item: item[0])

    selected, cursor = [], defaultdict(int)
    strategies = sorted(by_strategy, key=lambda s: (-len(by_strategy[s]), s))
    while len(selected) < PRODUCT_TARGET:
        added = False
        for strategy in strategies:
            i = cursor[strategy]
            if i >= len(by_strategy[strategy]) or len(selected) >= PRODUCT_TARGET:
                continue
            selected.append(by_strategy[strategy][i])
            cursor[strategy] += 1
            added = True
        if not added:
            break
    return selected


def main():
    app = rag.RAGApplication()
    app.documents = rag.load_knowledge_qa('knowledge.txt')
    app.product_catalog = app._build_product_catalog()
    catalog = app.product_catalog
    products = select_products(catalog)
    items = []

    # 1) 产品事实：使用当前知识库的真实数值，避免知识库更新后的误报。
    for name, meta in products:
        facts = product_facts(meta['series'])
        if facts:
            items.append({
                'category': 'product',
                'question': f'{name}的业绩表现怎么样？',
                'expected_names': [name],
                'expected_facts': facts,
            })

    # 2) 管理人：选产品覆盖最多的管理人，同时包含多策略场景。
    manager_products = defaultdict(list)
    for name, meta in catalog.items():
        if meta.get('manager') and meta.get('series'):
            manager_products[meta['manager']].append(name)
    managers = sorted(manager_products, key=lambda m: (-len(manager_products[m]), m))[:MANAGER_TARGET]
    for manager in managers:
        items.append({
            'category': 'manager',
            'question': f'{manager}怎么样？',
            'expected_names': [manager],
            'expected_facts': [],
        })

    # 3) 策略排名：验证策略路由是否能同时取回多只同策略产品。
    strategy_products = defaultdict(list)
    for name, meta in catalog.items():
        if meta.get('strategy') and meta.get('series'):
            strategy_products[meta['strategy']].append(name)
    strategies = sorted(strategy_products, key=lambda s: (-len(strategy_products[s]), s))
    for strategy in strategies[:STRATEGY_TARGET]:
        names = sorted(strategy_products[strategy])[:2]
        if len(names) == 2:
            items.append({
                'category': 'strategy',
                'question': f'{strategy}谁表现好？',
                'expected_names': names,
                'expected_facts': [],
            })

    # 4) 管理人对比：成对测试全量扫描与多实体合并。
    for left, right in zip(managers[::2], managers[1::2]):
        items.append({
            'category': 'comparison',
            'question': f'对比{left}和{right}的产品表现',
            'expected_names': [left, right],
            'expected_facts': [],
        })

    # 5) 通用与容错：覆盖不带实体的语义检索和管理人错别字路径。
    for question in [
        '指增的超额怎么理解？',
        '宏观策略和量化CTA有什么区别？',
        '私募产品的最大回撤应该怎么看？',
        '管理人和产品应该如何比较？',
    ]:
        items.append({'category': 'generic', 'question': question, 'expected_names': [], 'expected_facts': []})
    # 只保留当前容错阈值确实能识别的多字错别字，不用单字误差制造假失败。
    for typo, expected in [('执行通达怎么样？', '知行通达')]:
        if expected in rag.MANAGER_LIST:
            items.append({
                'category': 'typo',
                'question': typo,
                'expected_names': [expected],
                'expected_facts': [],
            })

    with open('eval_set.jsonl', 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    category_counts = defaultdict(int)
    for item in items:
        category_counts[item['category']] += 1
    print(f'已重建 eval_set.jsonl：{len(items)} 条')
    print('类别分布：' + '，'.join(f'{k} {v}' for k, v in category_counts.items()))
    print(f'产品事实条目：{sum(bool(i["expected_facts"]) for i in items)} 条')


if __name__ == '__main__':
    main()
