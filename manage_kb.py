"""知识库管理命令行：
  python manage_kb.py status              # 当前状态/版本/数据截止日期
  python manage_kb.py validate            # 数据质量校验
  python manage_kb.py rebuild --note x    # 校验后重建向量库（快照+审计）
  python manage_kb.py rollback <版本>      # 回滚 knowledge.txt 并重建
  python manage_kb.py log [n]             # 最近 n 条审计记录
"""

import argparse
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import kb_management as kbm


def _check_server_lock():
    """检测 7860 端口是否被运行中的服务占用（占用会导致 chroma 文件无法删除）。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', 7860))
        return False
    except OSError:
        return True
    finally:
        s.close()


def _require_server_stopped():
    if _check_server_lock():
        print('❌ 检测到应用服务正在运行（端口 7860），请先停止服务再执行此操作，'
              '否则向量库文件被占用会导致失败。')
        sys.exit(1)


def cmd_status(args):
    info = kbm.get_kb_info('knowledge.txt')
    print('知识库路径:', info['path'])
    print('当前 SHA256:', info['current_sha256'][:16], '...')
    print('文件大小:', info['current_size'])
    print('数据截止日期:', info['data_date'] or '未标注')
    if info['state']:
        s = info['state']
        print('上次构建:', s['built_at'], '| 文档数:', s['docs'])
        print('上次构建哈希:', s['file_sha256'][:16], '...')
    print('需要重建:', '是' if info['needs_rebuild'] else '否')
    print('历史版本数:', len(info['versions']))
    for v in info['versions'][:5]:
        print('  -', v['version'], f"({v['size']} 字节)")


def cmd_validate(args):
    report = kbm.validate_knowledge('knowledge.txt')
    st = report['stats']
    print(f"问答块数: {st['blocks']} | 无回答: {st['no_a']} | 产品数: {st['products']}")
    if not report['issues']:
        print('✅ 未发现数据质量问题')
        return
    for issue in report['issues']:
        print(f"⚠️  {issue['type']} x{issue.get('count', 1)}")
        for item in issue.get('items', [])[:5]:
            print('     ', item)
        if 'product' in issue:
            print('     产品:', issue.get('product'))


def cmd_rebuild(args):
    _require_server_stopped()
    print('== 数据质量校验 ==')
    cmd_validate(args)
    if args.strict:
        report = kbm.validate_knowledge('knowledge.txt')
        if report['issues']:
            print('❌ --strict 模式下发现质量问题，中止重建')
            sys.exit(1)
    persist = './chroma_db'
    if os.path.exists(persist):
        shutil.rmtree(persist)
    print('== 开始重建向量库 ==')
    import rag_graph业绩 as rag
    app = rag.RAGApplication()
    app.initialize('knowledge.txt')
    print('✅ 重建完成，文档数:', len(app.documents))
    if args.note:
        print('  备注:', args.note)


def cmd_rollback(args):
    _require_server_stopped()
    target = kbm.rollback(args.version)
    print('已回滚 knowledge.txt <-', target)
    persist = './chroma_db'
    if os.path.exists(persist):
        shutil.rmtree(persist)
    import rag_graph业绩 as rag
    app = rag.RAGApplication()
    app.initialize('knowledge.txt')
    print('✅ 回滚后向量库已重建，文档数:', len(app.documents))


def cmd_log(args):
    if not os.path.exists(kbm.AUDIT_FILE):
        print('暂无审计记录')
        return
    lines = open(kbm.AUDIT_FILE, encoding='utf-8').read().splitlines()
    for line in lines[-args.n:]:
        import json
        e = json.loads(line)
        keys = ['ts', 'action', 'note', 'docs', 'data_date', 'file_sha256', 'version']
        print('  '.join(f"{k}={e.get(k)}" for k in keys if e.get(k)))


def main():
    p = argparse.ArgumentParser(description='知识库管理')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status', help='查看状态').set_defaults(fn=cmd_status)
    sub.add_parser('validate', help='数据质量校验').set_defaults(fn=cmd_validate)
    rp = sub.add_parser('rebuild', help='重建向量库')
    rp.add_argument('--note', default='', help='变更备注')
    rp.add_argument('--strict', action='store_true', help='发现质量问题即中止')
    rp.set_defaults(fn=cmd_rebuild)
    rp2 = sub.add_parser('rollback', help='回滚到指定版本')
    rp2.add_argument('version', help='版本标识，如 20260806_183000')
    rp2.set_defaults(fn=cmd_rollback)
    lp = sub.add_parser('log', help='审计日志')
    lp.add_argument('n', nargs='?', type=int, default=10, help='条数')
    lp.set_defaults(fn=cmd_log)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
