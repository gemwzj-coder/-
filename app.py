import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Gradio 启动时会用 httpx 请求自己的 /gradio_api/startup-events 接口。
# 若 Windows/公司代理劫持 localhost，该请求可能被代理返回 502；强制本机地址直连。
_no_proxy_hosts = ['127.0.0.1', 'localhost', '::1']
_no_proxy = os.environ.get('NO_PROXY') or os.environ.get('no_proxy') or ''
_no_proxy_items = [item.strip() for item in _no_proxy.split(',') if item.strip()]
for _host in _no_proxy_hosts:
    if _host not in _no_proxy_items:
        _no_proxy_items.append(_host)
os.environ['NO_PROXY'] = ','.join(_no_proxy_items)
os.environ['no_proxy'] = os.environ['NO_PROXY']

import base64
import shutil
import sys
import gradio as gr
import kb_management as kbm
import evaluate_rag as evalg
from rag_graph业绩 import stream_answer, _app, analyze_client_portfolio, analyze_style

# 控制台/日志输出统一 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LULU_IMG_PATH = os.path.join(BASE_DIR, 'assets', 'lulu-capybara-v2.png')


def _image_data_uri(path: str) -> str:
    with open(path, 'rb') as f:
        content = f.read()
    return 'data:image/png;base64,' + base64.b64encode(content).decode('ascii')


LULU_IMG = _image_data_uri(LULU_IMG_PATH)
USER_AVATAR_PATH = os.path.join(BASE_DIR, 'assets', 'generic-user.svg')
USER_IMG = ''

try:
    _KB_INFO = kbm.get_kb_info('knowledge.txt')
    DATA_DATE = _KB_INFO.get('data_date') or '未知'
except Exception:
    DATA_DATE = '未知'

GREETING = (
    "噜噜来啦！🌼 我是噜噜，一只慢悠悠、但很懂量化投资的水豚。"
    "\n\n可以问我：管理人怎么样、策略怎么选、产品对比；我会把复杂的事讲得明白一点。"
    f"\n\n📅 我的知识库数据截至 {DATA_DATE}。"
)


def respond_stream(user_input, history):
    """Gradio 聊天入口：更新页面消息，并把 RAG 的流式片段逐步展示出来。"""
    if not user_input or not user_input.strip():
        yield history
        return

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": ""})

    try:
        for partial_answer in stream_answer(user_input, session_id="default"):
            history[-1]["content"] = partial_answer
            yield history

    except Exception as e:
        history[-1]["content"] = f"😔 抱歉，噜噜暂时无法回答这个问题。错误：{str(e)}"
        yield history


# ==================== 知识库管理（管理员面板） ====================

def kb_status_text():
    try:
        info = kbm.get_kb_info('knowledge.txt')
        state = info['state'] or {}
        lines = [
            f"- **知识库文件**：`knowledge.txt`（{info['current_size']} 字节）",
            f"- **文档数**：{state.get('docs', '未知')}",
            f"- **数据截止日期**：{info['data_date'] or '未标注'}",
            f"- **上次构建**：{state.get('built_at', '从未')}",
            f"- **需要重建**：{'是（点下方按钮或重启应用）' if info['needs_rebuild'] else '否'}",
            f"- **历史版本**：{len(info['versions'])} 个",
        ]
        for v in info['versions'][:5]:
            lines.append(f"  - `{v['version']}`")
        return '\n'.join(lines)
    except Exception as e:
        return f'读取状态失败：{e}'


def _rebuild_now():
    """在进程内重建向量库（关闭旧客户端 → 删除索引 → 重新构建）。"""
    try:
        try:
            _app.vectorstore._client.close()
        except Exception:
            pass
        persist = './chroma_db'
        if os.path.exists(persist):
            shutil.rmtree(persist)
        _app._initialized = False
        _app.initialize('knowledge.txt')
        return f'✅ 重建完成，当前文档数：{len(_app.documents)}，状态已更新。'
    except Exception as e:
        return f'❌ 重建失败：{e}\n（若提示文件被占用，请停止应用后重新启动，启动时会自动重建）'


def do_rebuild():
    return _rebuild_now()


def do_eval():
    try:
        items = evalg.load_eval_set('eval_set.jsonl') + evalg.load_eval_set('eval_bad_cases.jsonl')
        # 管理台默认跑无 LLM 的回归：检索和确定性合规护栏均可重复、无额外模型成本。
        report = evalg.run_eval(items, use_answers=False)
        a = report['aggregate']
        def pct(value):
            return f'{value:.1%}' if isinstance(value, (int, float)) else 'N/A'
        text = (
            f"**实体路由覆盖率**：{pct(a.get('route_entity_coverage_rate'))}（{report['n']} 条）\n"
            f"**产品向量 Top-1 / Top-3**："
            f"{pct(a.get('product_vector_top1_hit_rate'))} / {pct(a.get('product_vector_top3_hit_rate'))}\n"
            f"**合规拒答/转人工**：{pct(a.get('compliance_refusal_or_transfer_rate'))}\n"
            f"**平均检索耗时**：{a['avg_retrieval_s']}s\n"
            f"**事实一致性 / 引用完整性**：需运行 `python evaluate_rag.py --suite all --answers` 生成回答评估\n"
        )
        fails = [r['question'] for r in report['items'] if r.get('route_entity_coverage') is False]
        if fails:
            text += '\n**未命中**：\n' + '\n'.join(f'- {q}' for q in fails)
        else:
            text += '\n全部命中 🎉'
        return text
    except Exception as e:
        return f'❌ 评估失败：{e}'


def do_rollback(version):
    if not version:
        return '请先在下拉框选择一个版本'
    try:
        kbm.rollback(version)
        return _rebuild_now() + f'\n（已回滚到版本 {version}）'
    except Exception as e:
        return f'❌ 回滚失败：{e}'


# ==================== 水豚噜噜主题 ====================
CSS = """
:root {
    --lulu-cream: #FFF9F0;
    --lulu-card: #FFFFFF;
    --lulu-brown: #A9714B;
    --lulu-brown-dark: #8A5A3B;
    --lulu-pink: #FFB3A7;
    --lulu-peach: #FFD9A0;
    --lulu-mint: #A9D6C3;
    --lulu-text: #5C4633;
    --lulu-text-light: #9A7B62;
}

body {
    background:
        radial-gradient(circle at 12% 18%, rgba(255, 217, 160, 0.35) 0, rgba(255, 217, 160, 0) 28%),
        radial-gradient(circle at 88% 12%, rgba(255, 179, 167, 0.28) 0, rgba(255, 179, 167, 0) 26%),
        radial-gradient(circle at 80% 88%, rgba(169, 214, 195, 0.30) 0, rgba(169, 214, 195, 0) 30%),
        linear-gradient(160deg, #FFF6EA 0%, #FFF0E0 100%);
    background-attachment: fixed;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 26px 22px 42px !important;
    background: rgba(255, 249, 240, 0.86) !important;
    border: 2px solid #F7E3CC !important;
    border-radius: 30px !important;
    box-shadow: 0 18px 48px rgba(169, 113, 75, 0.16) !important;
}

/* ===== 顶部：水豚噜噜 ===== */
.lulu-header {
    text-align: center;
    padding: 14px 0 10px 0;
    margin-bottom: 16px;
}

.lulu-mascot {
    position: relative;
    display: inline-block;
    margin-bottom: 6px;
}

.lulu-img {
    width: 128px;
    height: 128px;
    border-radius: 50%;
    box-shadow: 0 10px 24px rgba(169, 113, 75, 0.28);
    background: #FFFDF8;
    animation: lulu-float 3.4s ease-in-out infinite;
}

.lulu-bubble {
    position: absolute;
    top: -14px;
    right: -128px;
    background: #FFFFFF;
    color: var(--lulu-text);
    border: 2px solid #F4DCC2;
    border-radius: 16px;
    padding: 7px 14px;
    font-size: 0.92rem;
    font-weight: 600;
    white-space: nowrap;
    box-shadow: 0 6px 16px rgba(169, 113, 75, 0.14);
    animation: lulu-pop 0.7s ease both;
}

.lulu-bubble::after {
    content: "";
    position: absolute;
    left: -9px;
    top: 22px;
    border: 7px solid transparent;
    border-right-color: #FFFFFF;
    transform: rotate(-20deg);
}

.lulu-title {
    font-size: 2.35rem;
    font-weight: 800;
    margin: 6px 0 2px 0;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #A9714B, #E08A5E, #D96A76);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.lulu-sub {
    color: var(--lulu-text-light);
    font-size: 0.98rem;
    margin: 0;
}

/* ===== 聊天区 ===== */
.chatbot {
    border-radius: 22px !important;
    overflow: hidden !important;
    background: #FFFDF9 !important;
    border: 1.5px solid #F2E0CB !important;
    box-shadow: inset 0 2px 10px rgba(169, 113, 75, 0.05) !important;
}

.message-wrap { max-width: 86% !important; }

.message {
    padding: 10px 14px !important;
    border-radius: 18px !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    animation: lulu-fade 0.35s ease both !important;
}

.message.bot, .assistant-message {
    background: #FFFFFF !important;
    border: 1.5px solid #F1DCC4 !important;
    border-top-left-radius: 6px !important;
    color: var(--lulu-text) !important;
    box-shadow: 0 3px 10px rgba(169, 113, 75, 0.08) !important;
}

.message.user, .user-message {
    background: linear-gradient(135deg, #FFE6C9, #FFD6B6) !important;
    border: 1.5px solid #F6C79E !important;
    border-top-right-radius: 6px !important;
    color: #7A5033 !important;
    box-shadow: 0 3px 10px rgba(233, 150, 96, 0.12) !important;
}

.assistant-message img, .prose img {
    max-width: 100% !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
}

.prose { max-width: 100% !important; }

/* ===== 输入区 ===== */
.input-row { gap: 10px !important; margin-top: 14px !important; }

.input-row textarea {
    border-radius: 16px !important;
    border: 2px solid #F0D9BE !important;
    background: #FFFDF9 !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    color: var(--lulu-text) !important;
    transition: all 0.2s !important;
}

.input-row textarea:focus {
    border-color: #E89A6B !important;
    box-shadow: 0 0 0 4px rgba(232, 154, 107, 0.16) !important;
}

.send-btn {
    background: linear-gradient(135deg, #C98F62, #E8916B) !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    min-height: 50px !important;
    box-shadow: 0 6px 16px rgba(201, 143, 98, 0.35) !important;
    transition: all 0.18s !important;
}

.send-btn:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 10px 22px rgba(201, 143, 98, 0.45) !important;
}

.send-btn:active { transform: scale(0.96) !important; }

.clear-btn {
    border-radius: 999px !important;
    font-weight: 600 !important;
    border: 2px solid #F0D9BE !important;
    background: #FFFFFF !important;
    color: var(--lulu-text) !important;
    transition: all 0.18s !important;
}

.clear-btn:hover {
    background: #FFF2E3 !important;
    border-color: #E89A6B !important;
}

/* ===== 示例问题 ===== */
.examples-label {
    color: var(--lulu-text) !important;
    font-weight: 700 !important;
    margin-top: 18px !important;
    margin-bottom: 8px !important;
}

.examples-row button {
    border-radius: 999px !important;
    border: 2px solid #F2E0CB !important;
    padding: 8px 18px !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
    color: var(--lulu-text) !important;
    transition: all 0.18s !important;
}

.examples-row button:hover {
    border-color: #E89A6B !important;
    background: #FFF2E3 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(201, 143, 98, 0.18) !important;
}

/* ===== 提示 & 页脚 ===== */
.lulu-tip {
    background: #FFF3E0;
    border: 1.5px dashed #E9C79F;
    border-radius: 14px;
    padding: 8px 14px;
    color: #9A7B62;
    font-size: 0.85rem;
    margin-top: 14px;
}

.lulu-footer {
    text-align: center;
    color: #B99B82;
    font-size: 0.78rem;
    margin-top: 18px;
}

/* ===== 动效 ===== */
@keyframes lulu-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-9px); }
}

@keyframes lulu-pop {
    0% { opacity: 0; transform: scale(0.7); }
    70% { transform: scale(1.05); }
    100% { opacity: 1; transform: scale(1); }
}

@keyframes lulu-fade {
    0% { opacity: 0; transform: translateY(6px); }
    100% { opacity: 1; transform: translateY(0); }
}

.message img { cursor: zoom-in; }

/* ===== 噜噜：软糖色 3D 水豚主题覆盖 ===== */
:root {
    --lulu-cream: #FFF7F1;
    --lulu-card: rgba(255, 255, 255, .82);
    --lulu-brown: #C46F58;
    --lulu-brown-dark: #815047;
    --lulu-pink: #F7A2B9;
    --lulu-peach: #FFC88F;
    --lulu-mint: #BBDCCF;
    --lulu-text: #654B49;
    --lulu-text-light: #A47E7C;
}

body {
    background:
        radial-gradient(circle at 88% 6%, rgba(255, 220, 116, .62) 0 5%, rgba(255, 220, 116, 0) 26%),
        radial-gradient(circle at 9% 28%, rgba(188, 150, 230, .34) 0, rgba(188, 150, 230, 0) 29%),
        radial-gradient(circle at 90% 74%, rgba(255, 145, 177, .27) 0, rgba(255, 145, 177, 0) 30%),
        linear-gradient(145deg, #EEC6DE 0%, #FAD0BF 47%, #FFE6B8 100%);
}

.gradio-container {
    max-width: 920px !important;
    border: 1px solid rgba(255,255,255,.82) !important;
    border-radius: 36px !important;
    background: rgba(255, 249, 247, .74) !important;
    box-shadow: 0 24px 70px rgba(148, 81, 116, .25), inset 0 1px rgba(255,255,255,.9) !important;
    backdrop-filter: blur(18px);
}

.lulu-header {
    padding: 8px 0 12px !important;
    margin-bottom: 10px !important;
}

.lulu-mascot { margin-bottom: -12px !important; }

.lulu-img {
    width: 250px !important;
    height: 250px !important;
    object-fit: cover !important;
    object-position: center !important;
    border-radius: 30px !important;
    border: 4px solid rgba(255, 255, 255, .62) !important;
    background: transparent !important;
    box-shadow: 0 14px 26px rgba(142, 79, 97, .19) !important;
    filter: drop-shadow(0 14px 12px rgba(142, 79, 97, .22));
}

.lulu-bubble {
    top: 8px !important;
    right: -154px !important;
    border: 0 !important;
    color: #76545B !important;
    background: rgba(255,255,255,.88) !important;
    border-radius: 18px !important;
    box-shadow: 0 9px 20px rgba(145, 85, 110, .15) !important;
}

.lulu-title {
    font-size: 2.55rem !important;
    letter-spacing: 1px !important;
    background: linear-gradient(125deg, #E8798F, #D66B73 45%, #F19E55) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
}

.lulu-sub { font-size: 1rem !important; color: #9E747A !important; }

.chatbot {
    border: 1px solid rgba(255,255,255,.95) !important;
    border-radius: 26px !important;
    background: rgba(255,255,255,.58) !important;
    box-shadow: 0 10px 30px rgba(160, 93, 116, .10), inset 0 1px rgba(255,255,255,.85) !important;
}

.message.bot, .assistant-message {
    border: 0 !important;
    background: rgba(255,255,255,.93) !important;
    color: #634B4A !important;
    box-shadow: 0 5px 14px rgba(146, 91, 102, .10) !important;
}

.message.user, .user-message {
    border: 0 !important;
    background: linear-gradient(135deg, #FFB0C4, #F594A9) !important;
    color: #6E3844 !important;
    box-shadow: 0 6px 15px rgba(229, 112, 145, .20) !important;
}

.input-row textarea {
    border: 1px solid rgba(247, 165, 185, .48) !important;
    background: rgba(255,255,255,.82) !important;
    border-radius: 19px !important;
}
.input-row textarea:focus {
    border-color: #ED91AA !important;
    box-shadow: 0 0 0 4px rgba(237,145,170,.16) !important;
}
.send-btn {
    background: linear-gradient(135deg, #F18FA9, #E86F8C) !important;
    box-shadow: 0 8px 18px rgba(226, 103, 140, .30) !important;
}
.examples-row button, .clear-btn {
    border: 1px solid rgba(238, 158, 179, .38) !important;
    background: rgba(255,255,255,.76) !important;
    color: #80545A !important;
}
.examples-row button:hover, .clear-btn:hover {
    border-color: #E889A5 !important;
    background: #FFF0F4 !important;
}
.lulu-tip {
    background: rgba(255, 238, 209, .70) !important;
    border: 1px dashed rgba(224, 155, 110, .55) !important;
    color: #956E62 !important;
}

@media (max-width: 640px) {
    .gradio-container { padding: 18px 12px 30px !important; border-radius: 24px !important; }
    .lulu-img { width: 190px !important; height: 190px !important; }
    .lulu-bubble { position: static !important; display: inline-block !important; margin: -8px 0 8px !important; }
    .lulu-bubble::after { display: none; }
    .lulu-title { font-size: 2.2rem !important; }
}

/* ===== Internal Copilot professional override ===== */
:root { color-scheme: dark; }
body {
    background: #0f1117 !important;
    color: #e7eaf0 !important;
}
.gradio-container {
    max-width: 1180px !important;
    min-height: calc(100vh - 44px) !important;
    margin: 22px auto !important;
    padding: 22px 26px 30px !important;
    background: #171a21 !important;
    border: 1px solid #2a2f3a !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 70px rgba(0,0,0,.34) !important;
    backdrop-filter: none !important;
}
.workbench-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 18px;
    margin-bottom: 14px;
    border-bottom: 1px solid #2a2f3a;
}
.workbench-brand { display: flex; align-items: center; gap: 12px; }
.assistant-avatar {
    width: 72px; height: 72px; object-fit: cover; border-radius: 14px;
    border: 1px solid #556175; background: #222734;
    box-shadow: 0 8px 22px rgba(0,0,0,.28);
}
.workbench-brand h1 {
    margin: 0; color: #f5f7fb; font: 650 17px/1.25 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    letter-spacing: -.2px;
}
.workbench-brand p { margin: 3px 0 0; color: #8f98a8; font-size: 12px; }
.workbench-status { color: #9da7b8; font-size: 12px; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #31c48d; margin-right: 6px; box-shadow: 0 0 0 3px rgba(49,196,141,.12); }
.status-sep { margin: 0 9px; color: #4d5666; }
.gradio-container .tab-nav { gap: 4px !important; border-bottom: 1px solid #2a2f3a !important; margin-bottom: 18px !important; }
.gradio-container .tab-nav button {
    color: #9aa4b5 !important; background: transparent !important; border: 0 !important;
    border-radius: 8px 8px 0 0 !important; padding: 10px 14px !important; font-weight: 560 !important;
}
.gradio-container .tab-nav button.selected { color: #f2f6ff !important; background: #222733 !important; box-shadow: inset 0 -2px #6aa7ff !important; }
.gradio-container .tab-nav button:hover { color: #f2f6ff !important; background: #202530 !important; }
.gradio-container [role="tab"] { color: #c5cedb !important; }
.gradio-container [role="tab"][aria-selected="true"] { color: #ffffff !important; background: #222b38 !important; }
.chatbot { background: #11141b !important; border: 1px solid #2a2f3a !important; border-radius: 12px !important; box-shadow: none !important; }
.message.bot, .assistant-message { background: #202632 !important; border: 1px solid #303849 !important; color: #e6ebf4 !important; border-radius: 10px !important; box-shadow: none !important; }
.message.user, .user-message { background: #1d4e89 !important; color: #f4f8ff !important; border: 1px solid #2b68ad !important; border-radius: 10px !important; box-shadow: none !important; }
.chatbot .message, .chatbot .message *, .chatbot .assistant-message, .chatbot .assistant-message * { color: #edf2fa !important; opacity: 1 !important; }
.chatbot .message.user, .chatbot .message.user *, .chatbot .user-message, .chatbot .user-message * { color: #ffffff !important; }
.input-row textarea { background: #11141b !important; color: #edf1f7 !important; border: 1px solid #323949 !important; border-radius: 10px !important; }
.input-row textarea:focus { border-color: #6aa7ff !important; box-shadow: 0 0 0 3px rgba(106,167,255,.14) !important; }
.send-btn { background: #2f74c0 !important; border: 1px solid #4387d2 !important; border-radius: 9px !important; box-shadow: none !important; min-height: 48px !important; }
.send-btn:hover { background: #3982d3 !important; transform: none !important; box-shadow: none !important; }
.clear-btn, .examples-row button {
    background: #1d222c !important; color: #c3cad6 !important; border: 1px solid #333b4b !important; border-radius: 8px !important;
}
.examples-row button:hover, .clear-btn:hover { background: #262d39 !important; border-color: #48546a !important; transform: none !important; box-shadow: none !important; }
.lulu-tip { background: #151c28 !important; border: 1px solid #2c405d !important; color: #9fb6d5 !important; border-radius: 9px !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 { color: #f5f7fb !important; }
.gradio-container .prose, .gradio-container .markdown, .gradio-container .prose p, .gradio-container .markdown p { color: #c8d0dc !important; opacity: 1 !important; }
.gradio-container input, .gradio-container textarea { color-scheme: dark; }

/* ===== Modao-inspired three-column workspace ===== */
:root { color-scheme: light !important; }
html, body { background: #eef1f5 !important; color: #1d2939 !important; }
.gradio-container {
    position: relative !important;
    box-sizing: border-box !important;
    max-width: 1420px !important;
    min-height: calc(100vh - 32px) !important;
    margin: 16px auto !important;
    padding: 26px 278px 32px 270px !important;
    background: #ffffff !important;
    border: 1px solid #e4e9f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 34px rgba(31, 50, 81, .12) !important;
}
.app-sidebar {
    position: absolute !important; inset: 0 auto 0 0 !important; z-index: 4 !important;
    width: 238px !important; box-sizing: border-box !important;
    display: flex !important; flex-direction: column !important;
    padding: 24px 16px 18px !important;
    background: #fbfcfe !important; border-right: 1px solid #e7ecf3 !important;
    border-radius: 14px 0 0 14px !important;
    color: #324258 !important;
}
.sidebar-logo { width: 34px !important; height: 34px !important; display: grid !important; place-items: center !important; border-radius: 10px !important; background: linear-gradient(145deg,#6366f1,#3b5ccc) !important; color: #fff !important; font-weight: 800 !important; font-size: 18px !important; }
.sidebar-name { margin: 13px 0 18px !important; color: #182338 !important; font-weight: 750 !important; font-size: 15px !important; }
.sidebar-search { display:flex !important; gap:8px !important; align-items:center !important; padding: 10px 11px !important; margin-bottom: 20px !important; border: 1px solid #dfe6ef !important; border-radius: 10px !important; background:#fff !important; color:#8b98aa !important; font-size:12px !important; }
.sidebar-section { margin: 0 9px 8px !important; color: #9aa7b8 !important; font-size: 11px !important; letter-spacing: .06em !important; }
.sidebar-nav { width:100% !important; display:flex !important; align-items:center !important; gap:9px !important; padding:10px 10px !important; border:0 !important; border-radius:9px !important; background:transparent !important; color:#596a81 !important; text-align:left !important; font-size:13px !important; cursor:pointer !important; }
.sidebar-nav:hover, .sidebar-nav.active { background:#edf2ff !important; color:#365ac6 !important; font-weight:650 !important; }
.sidebar-rule { height:1px !important; background:#e9edf3 !important; margin:19px 4px !important; }
.sidebar-muted { padding:8px 10px !important; color:#73829a !important; font-size:12px !important; }
.sidebar-profile { display:flex !important; align-items:center !important; gap:9px !important; margin-top:auto !important; padding:12px 9px 0 !important; border-top:1px solid #e9edf3 !important; }
.sidebar-profile img { width:34px !important; height:34px !important; border-radius:50% !important; object-fit:cover !important; border:2px solid #fff !important; box-shadow:0 1px 4px rgba(20,40,70,.18) !important; }
.sidebar-profile b { display:block !important; color:#26354b !important; font-size:12px !important; }.sidebar-profile small { color:#8c99aa !important; font-size:10px !important; }
.source-panel { position:absolute !important; inset:0 0 0 auto !important; z-index:3 !important; width:252px !important; box-sizing:border-box !important; padding:28px 18px !important; background:#fff !important; border-left:1px solid #e7ecf3 !important; border-radius:0 14px 14px 0 !important; color:#334155 !important; }
.source-title { display:flex !important; justify-content:space-between !important; align-items:center !important; color:#1e2a3b !important; font-size:14px !important; font-weight:750 !important; margin-bottom:16px !important; }.source-title span { color:#5967e9 !important; }
.source-card { padding:13px 12px !important; margin-bottom:10px !important; border:1px solid #e5eaf1 !important; border-radius:10px !important; background:#fbfcfe !important; }.source-card b { color:#334155 !important; font-size:12px !important; }.source-card p { margin:6px 0 0 !important; color:#718096 !important; font-size:11px !important; line-height:1.55 !important; }
.source-actions { display:flex !important; justify-content:space-between !important; padding:13px 2px !important; border-bottom:1px solid #edf0f4 !important; color:#758399 !important; font-size:11px !important; }.source-actions b { color:#4557bf !important; font-size:11px !important; }
.workbench-header { padding:0 0 18px !important; margin-bottom:10px !important; border-color:#e8edf3 !important; }.workbench-brand { gap:10px !important; }.assistant-avatar { width:52px !important; height:52px !important; border-radius:50% !important; border:2px solid #fff !important; background:#f6f7ff !important; box-shadow:0 2px 8px rgba(40,61,101,.18) !important; }.workbench-brand h1 { color:#19263a !important; font-size:17px !important; }.workbench-brand p,.workbench-status { color:#73829a !important; }.status-sep { color:#c1cad7 !important; }
.gradio-container .tab-nav { border-color:#e8edf3 !important; margin-bottom:15px !important; }.gradio-container .tab-nav button { color:#718096 !important; }.gradio-container .tab-nav button.selected,.gradio-container [role="tab"][aria-selected="true"] { color:#3e55ce !important; background:#f5f6ff !important; box-shadow:inset 0 -2px #5865e9 !important; }.gradio-container .tab-nav button:hover { color:#3e55ce !important; background:#f5f6ff !important; }
.gradio-container h1,.gradio-container h2,.gradio-container h3,.gradio-container h4 { color:#1e293b !important; }.gradio-container .prose,.gradio-container .markdown,.gradio-container .prose p,.gradio-container .markdown p { color:#64748b !important; }
.chatbot { background:#fff !important; border:1px solid #e4e9f0 !important; border-radius:12px !important; }.message.bot,.assistant-message { background:#f2f5fa !important; border:1px solid #dbe3ee !important; color:#1f2d3d !important; border-radius:12px !important; }.message.user,.user-message { background:#426fce !important; border:1px solid #345fbb !important; color:#fff !important; border-radius:12px !important; }.chatbot .message,.chatbot .message *,.chatbot .assistant-message,.chatbot .assistant-message * { color:#1f2d3d !important; opacity:1 !important; }.chatbot .message.user,.chatbot .message.user *,.chatbot .user-message,.chatbot .user-message * { color:#fff !important; }
.input-row textarea { background:#fff !important; color:#243247 !important; border-color:#d9e1ec !important; }.input-row textarea:focus { border-color:#6574e8 !important; box-shadow:0 0 0 3px rgba(101,116,232,.13) !important; }.send-btn { background:#4f5ee7 !important; border-color:#4f5ee7 !important; }.send-btn:hover { background:#414fce !important; }.clear-btn,.examples-row button { background:#fff !important; color:#51627b !important; border-color:#dbe3ee !important; }.lulu-tip { background:#f7f9fc !important; color:#66768d !important; border-color:#dce5ef !important; }.gradio-container input,.gradio-container textarea { color-scheme:light !important; }

/* Keep the first release deliberately simple: one stable content column. */
.app-sidebar, .source-panel { display: none !important; }
.gradio-container {
    max-width: 1120px !important;
    min-height: calc(100vh - 32px) !important;
    margin: 16px auto !important;
    padding: 28px 34px 34px !important;
}
.workbench-header { margin-bottom: 18px !important; }
.chatbot { min-height: 480px !important; }

@media (max-width: 1060px) { .gradio-container { padding-right:28px !important; } .source-panel { display:none !important; } }
@media (max-width: 760px) { .gradio-container { margin:0 !important; padding:18px !important; border-radius:0 !important; } .app-sidebar { display:none !important; } .workbench-header { padding-top:0 !important; } }
@media (max-width: 640px) {
    .gradio-container { margin: 0 !important; min-height: 100vh !important; border-radius: 0 !important; padding: 16px !important; }
    .workbench-header { align-items: flex-start; gap: 12px; flex-direction: column; }
}
"""

# 前端交互：Enter=发送 / Shift+Enter=换行；图表图片点击放大。
# 注意：gradio 会把 load(js=...) 包装成 `await (js)(...args)` 编译，
# 所以这里必须是"函数表达式"（箭头函数），不能是立即执行语句，也不能以分号结尾。
_UI_JS = """
() => {
  if (window.__lulu_ui_setup) return;
  window.__lulu_ui_setup = true;

  // ---- Enter=发送，Shift+Enter=换行 ----
  document.addEventListener('keydown', function(e){
    if (e.key !== 'Enter') return;
    var box = document.querySelector('#msg-input-id textarea');
    if (!box || document.activeElement !== box) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.shiftKey) {
      var s = box.selectionStart, en = box.selectionEnd;
      box.value = box.value.slice(0, s) + '\\n' + box.value.slice(en);
      box.selectionStart = box.selectionEnd = s + 1;
      box.dispatchEvent(new Event('input', {bubbles: true}));
    } else {
      var btn = document.getElementById('send-btn-id');
      if (!btn) {
        var all = document.querySelectorAll('button');
        for (var i = 0; i < all.length; i++) {
          if (all[i].textContent.indexOf('发送') >= 0) { btn = all[i]; break; }
        }
      }
      if (btn) btn.click();
    }
  }, true);

  // ---- 图片点击放大 ----
  var overlay = document.createElement('div');
  overlay.id = 'lulu-zoom';
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;' +
    'background:rgba(0,0,0,0.82);display:none;align-items:center;justify-content:center;' +
    'z-index:99999;cursor:zoom-out;';
  var zimg = document.createElement('img');
  zimg.style.cssText = 'max-width:94vw;max-height:94vh;border-radius:10px;' +
    'box-shadow:0 10px 40px rgba(0,0,0,0.6);';
  overlay.appendChild(zimg);
  document.body.appendChild(overlay);
  function closeZoom(){ overlay.style.display = 'none'; }
  overlay.addEventListener('click', closeZoom);
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') closeZoom(); });
  document.addEventListener('click', function(e){
    var t = e.target;
    if (t && t.tagName === 'IMG' && t.src
        && (t.src.indexOf('data:image') === 0 || t.src.indexOf('/gradio_api/file=') >= 0)
        && t.closest && t.closest('.message')) {
      zimg.src = t.src;
      overlay.style.display = 'flex';
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);
}
"""


# ==================== 界面 ====================
# CSS 已在 launch() 中传入；不要在 Blocks 和 launch() 重复传，兼容 Gradio 6。
with gr.Blocks(title="私募研究与运营助手 · Internal Research Copilot") as demo:
    gr.HTML(f"""
        <header class="workbench-header">
            <div class="workbench-brand">
                <img class="assistant-avatar" src="{LULU_IMG}" alt="助手头像">
                <div>
                    <h1>私募研究与运营助手</h1>
                    <p>Internal Research Copilot · Demo Environment</p>
                </div>
            </div>
            <div class="workbench-status">
                <span class="status-dot"></span> 知识库已连接
                <span class="status-sep">•</span> 仅供内部使用
            </div>
        </header>
    """)
    gr.HTML(f"""
        <aside class="app-sidebar">
          <div class="sidebar-logo">S</div>
          <div class="sidebar-name">私募研究助手</div>
          <div class="sidebar-search">⌕ <span>搜索工作区</span></div>
          <div class="sidebar-section">工作区</div>
          <button class="sidebar-nav active" onclick="document.querySelector('[role=tab]').click()">⌂ <span>产品检索与对比</span></button>
          <button class="sidebar-nav" onclick="Array.from(document.querySelectorAll('[role=tab]')).find(x=>x.innerText.includes('组合')).click()">◈ <span>组合 / 风格分析</span></button>
          <button class="sidebar-nav" onclick="Array.from(document.querySelectorAll('[role=tab]')).find(x=>x.innerText.includes('管理员')).click()">⚙ <span>知识库管理</span></button>
          <div class="sidebar-rule"></div>
          <div class="sidebar-section">快捷任务</div>
          <div class="sidebar-muted">产品与基准对比</div>
          <div class="sidebar-muted">管理人资料检索</div>
          <div class="sidebar-muted">合规初稿生成</div>
          <div class="sidebar-profile">
            <img src="{USER_IMG}" alt="用户头像"><div><b>吴泽杰</b><small>研究与运营</small></div>
          </div>
        </aside>
        <aside class="source-panel">
          <div class="source-title">工作台说明 <span>ⓘ</span></div>
          <div class="source-card"><b>内部资料检索</b><p>管理人、策略、产品与基准的结构化检索。</p></div>
          <div class="source-card"><b>可追溯输出</b><p>回答展示数据截止日期与检索依据。</p></div>
          <div class="source-card"><b>合规边界</b><p>不提供买卖时点、仓位或收益承诺。</p></div>
          <div class="source-actions"><span>知识库版本</span><b>{DATA_DATE}</b></div>
          <div class="source-actions"><span>评测与回归</span><b>管理员 Tab →</b></div>
        </aside>
    """)
    with gr.Tabs():
      with gr.Tab("产品检索与对比", id="search"):
        gr.Markdown("#### 内部资料检索工作台\n用于产品、管理人、策略与基准的事实检索和客观对比；不提供个性化投资建议。")
        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": GREETING}], label=None,
            elem_classes="chatbot", height=560, show_label=False, avatar_images=(USER_AVATAR_PATH, LULU_IMG_PATH), render_markdown=True)
        with gr.Row(elem_classes="input-row"):
            msg = gr.Textbox(placeholder="例如：对比衍复和幻方的 500 指增", lines=2, scale=5,
                show_label=False, container=False, elem_id="msg-input-id")
            send_btn = gr.Button("检索并生成初稿", variant="primary", elem_classes="send-btn", scale=1, elem_id="send-btn-id")
        with gr.Row():
            clear_btn = gr.Button("清空对话", elem_classes="clear-btn", scale=0)
            gr.Markdown("<span style='color:#B99B82;font-size:.82rem'>Enter 发送 · Shift+Enter 换行</span>")
        gr.HTML('<div class="lulu-tip">数据来自内部知识库；回答会展示检索依据。历史业绩不代表未来表现，不构成投资建议。</div>')
        with gr.Row(elem_classes="examples-row"):
            examples = ["对比衍复和幻方的500指增", "衍复小市值怎么样？", "私募产品的最大回撤应该怎么看？"]
            for ex in examples:
                gr.Button(ex, size="sm", variant="secondary").click(lambda x=ex: gr.update(value=x), None, msg)

      with gr.Tab("组合 / 风格分析", id="analysis"):
        gr.Markdown("#### 组合与风格分析\n面向内部运营与投顾支持，用于生成组合对比和风格漂移观察。")
        gr.Markdown(
            "把您关注的产品写进来（**不用写比例**，系统按等权估算组合），"
            "再选一个对比基准，就能看到**组合 vs 基准**的对比图。"
        )
        portfolio_input = gr.Textbox(
            value="远澜CTA、衍复1000指增、幻方500指增",
            label="输入 2-3 只产品（用、或+分隔，不用写比例）",
            lines=2,
        )
        portfolio_btn = gr.Button("🔍 生成组合对比", variant="primary")
        bench_radio = gr.Radio(
            choices=[],
            label="对比基准（点选即可切换对比图）",
            visible=False,
        )
        portfolio_out = gr.Markdown("")

        def portfolio_tab_analyze(question, selected):
            md, options, default = analyze_client_portfolio(question, selected)
            if md is None:
                return (
                    "😔 暂时无法生成分析：请确认输入了至少 2 只产品"
                    "（如：远澜CTA、衍复A500、幻方1000）。",
                    gr.update(choices=[], visible=False),
                )
            # 返回的是数据缺失提示时，没有可选基准；不显示空的单选框。
            if not options:
                return md, gr.update(choices=[], value=None, visible=False)
            keys = [k for _, k in options]
            value = selected if selected in keys else default
            return md, gr.update(choices=options, value=value, visible=True, label="对比基准（点选即可切换对比图）")

        portfolio_btn.click(
            portfolio_tab_analyze,
            [portfolio_input, bench_radio],
            [portfolio_out, bench_radio],
        )
        bench_radio.select(
            portfolio_tab_analyze,
            [portfolio_input, bench_radio],
            [portfolio_out, bench_radio],
        )

        gr.Markdown("#### 风格体检")
        gr.Markdown(
            "输入管理人（如：幻方）或产品名，查看旗下产品与各宽基指数的对比，"
            "判断有没有**风格漂移**（比如挂着1000指增、实际风格却偏向500）。"
        )
        style_input = gr.Textbox(
            value="幻方",
            label="管理人 / 产品名",
            lines=1,
        )
        style_btn = gr.Button("🔍 生成风格对比", variant="primary")
        style_product = gr.Radio(
            choices=[],
            label="选择产品（每只产品单独看）",
            visible=False,
        )
        style_idx = gr.Radio(
            choices=[],
            label="细分：选一个指数看两线详细对比",
            visible=False,
        )
        style_out = gr.Markdown("")

        def style_tab_analyze(query, product_selected, index_selected):
            md, prod_opts, default_prod, idx_opts, default_idx = analyze_style(
                query, product_selected, index_selected
            )
            if md is None:
                return (
                    "😔 未找到该管理人/产品的业绩数据，请检查名称（如：幻方、远澜红枫1号）。",
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                )
            # 管理人存在、但没有产品业绩数据时展示明确提示，并隐藏无效控件。
            if not prod_opts:
                return (
                    md,
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=[], value=None, visible=False),
                )
            prod_keys = [k for _, k in prod_opts]
            idx_keys = [k for _, k in idx_opts]
            pv = product_selected if product_selected in prod_keys else default_prod
            iv = index_selected if index_selected in idx_keys else default_idx
            return (
                md,
                gr.update(choices=prod_opts, value=pv, visible=len(prod_opts) > 1,
                          label="选择产品（每只产品单独看）"),
                gr.update(choices=idx_opts, value=iv, visible=True,
                          label="细分：选一个指数看两线详细对比"),
            )

        style_btn.click(
            style_tab_analyze,
            [style_input, style_product, style_idx],
            [style_out, style_product, style_idx],
        )
        style_product.select(
            style_tab_analyze,
            [style_input, style_product, style_idx],
            [style_out, style_product, style_idx],
        )
        style_idx.select(
            style_tab_analyze,
            [style_input, style_product, style_idx],
            [style_out, style_product, style_idx],
        )

      with gr.Tab("管理员", id="admin"):
        gr.Markdown("#### 知识库治理与评测\n在此查看版本状态、重建索引、运行回归评测与执行回滚。")
        gr.Markdown("直接编辑 `knowledge.txt` 保存后，点「重建向量库」即可生效；"
                    "想回到旧版本就从下拉框选一个再点「回滚并重建」。")
        kb_status = gr.Markdown(value=kb_status_text())
        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")
            rebuild_btn = gr.Button("♻️ 重建向量库", variant="primary")
            eval_btn = gr.Button("📊 运行检索评估", variant="secondary")
        with gr.Row():
            rollback_dd = gr.Dropdown(
                choices=[v['version'] for v in kbm.list_versions()],
                label="回滚到版本",
                scale=3,
            )
            rollback_btn = gr.Button("⏪ 回滚并重建", scale=1)
        admin_out = gr.Markdown("")

    send_btn.click(respond_stream, [msg, chatbot], chatbot).then(lambda: "", None, msg)
    msg.submit(respond_stream, [msg, chatbot], chatbot).then(lambda: "", None, msg)
    clear_btn.click(lambda: [{"role": "assistant", "content": GREETING}], None, chatbot)
    refresh_btn.click(kb_status_text, None, kb_status)
    rebuild_btn.click(do_rebuild, None, admin_out)
    eval_btn.click(do_eval, None, admin_out)
    rollback_btn.click(do_rollback, rollback_dd, admin_out)

    demo.load(js=_UI_JS)


if __name__ == "__main__":
    from rag_graph业绩 import _app
    print("🔄 正在预加载模型，请稍候...")
    _app.initialize('knowledge.txt')
    print("🚀 启动 Gradio 界面...")
    demo.launch(
        share=False,
        # 仅供本机使用。避免 Windows 上将 0.0.0.0 转为 localhost 时
        # 被本机代理/网络组件拦截，导致 Gradio 启动自检返回 502。
        server_name="127.0.0.1",
        # 默认从 7860 开始寻找可用端口；上次实例尚未退出时自动改用下一端口。
        server_port=None,
        ssr_mode=False,
        allowed_paths=[os.path.join(BASE_DIR, 'charts'), os.path.join(BASE_DIR, 'assets')],
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="rose",
            neutral_hue="neutral",
        ),
        css=CSS,
    )
