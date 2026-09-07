"""基金知识库问答的核心模块。

阅读路线（从上到下）：
1. 配置、名单与关键词：定义模型、知识库和查询识别所需的固定信息；
2. 数据解析与图表：从知识库文本中提取业绩、指数和产品序列；
3. ``RAGApplication``：初始化向量库、检索资料、组装 LangGraph 对话流程；
4. 文件末尾的公开函数：供 ``app.py`` 调用的问答、组合分析和风格分析入口。

本次整理只抽取重复代码并补充说明，不改变原有检索和回答路线。
"""

# --- 导入和配置 ---
import re
import os
import platform
import shutil
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load an ignored .env file for local development. Process-level environment
# variables remain the preferred configuration for deployments.
load_dotenv()

# 控制台/日志输出统一 UTF-8，避免 GBK 环境下的乱码或崩溃
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import matplotlib

# 在导入 pyplot 前指定无界面后端，避免 Windows 服务环境弹出图形窗口。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# app.py 可自行指定国内镜像；独立运行本模块时才使用默认地址。
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

from typing import TypedDict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- API 配置 ---
# 服务地址允许通过环境变量覆盖；密钥仅从本机的 local_config.py 读取。
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")

try:
    # 本地私用配置：文件已加入 .gitignore，避免误上传密钥。
    from local_config import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = ""

# --- 共享常量 ---
_env_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if _env_deepseek_api_key:
    DEEPSEEK_API_KEY = _env_deepseek_api_key

MANAGER_LIST = list(dict.fromkeys([
    '衍复', '幻方', '九坤', '宽德', '黑翼', '灵均', '鸣石', '平方和',
    '世纪前沿', '托特', '磐松', '子午', '稳博', '茂源', '诚奇', '天演',
    '明汯', '龙旗', '金锝', '千象', '因诺', '启林', '半鞅', '量派',
    '知行通达', '鲁民投', '洛书', '大岩', '白鹭', '乾象', '橡木', '同温层',
    '正仁', '正定', '正瀛', '盛冠达', '盛丰', '赫富', '超量子', '云起',
    '量锐', '量魁', '量道', '量盈', '衍盛', '衍合', '艾方', '博普',
    '合骥', '图灵', '天算', '玄信', '孝庸', '守正用奇', '安子', '安贤',
    '念觉', '思勰', '悦海盈和', '星阔', '鹿秀', '微观博易',
    '德贝', '弈倍', '盖亚青柯', '致远', '千朔', '千衍',
    '大道', '凡二', '信弘天禾', '兆信', '元图',
    '交睿', '众量', '优美利', '保银', '倍漾', '华年', '卓识', '嘉懿萃',
    '均成', '宁水', '守朴', '宏锡', '岚湖', '明毅', '涵德', '煜德',
    '盛泉恒元', '睿量', '纽达', '聚宽', '致诚卓远', '蒙玺', '衍盛铭量',
    '进化论', '远和', '远澜', '锐天', '阿巴马', '顽岩', '香农', '鸣熙',
    '仁布', '水璞', '源乐晟', 'HH',
    '喜岳', '日斗', '泓湖', '杭州波粒二象', '清溪', '思源',
    '高毅', '景林', '永安国富', '弘尚资产', '仲阳天王星', '博润', '乐道', '远信', '凯丰',
    'BX', '万方', '万柏', '世诚', '东方港湾', '中欧瑞博', '中泰', '中睿合银', '丹羿',
    '久期', '亘曦', '仁桥', '仙人掌', '会世', '信璞', '元胜', '兴聚', '冲积', '勤辰',
    '华安合鑫', '博润银泰', '博衍', '双隆', '合晟', '合远', '同犇', '名禹', '君之健',
    '和谐汇一', '嘉越', '国源信达', '壹点纳锦', '复胜', '大禾', '宁泉', '宽远',
    '少薮派', '展博', '展弘', '巨杉', '希瓦', '庄贤', '康曼德', '开思', '弘尚',
    '彤源', '思瑞', '慎知', '拾贝', '数法', '文多', '文谛', '新思哲', '无量', '明世',
    '明河', '易同', '朋锦仲阳', '望正', '林园', '枫泉', '正圆', '毕盛', '汉和',
    '泓澄', '洪运瑞恒', '涌津', '淡水泉', '清和泉', '玄元', '玖鹏', '瓴仁', '申九',
    '盘京', '睿扬', '睿璞', '睿远', '睿郡', '石锋', '磐耀', '竹润', '细水', '翰荣',
    '翼虎', '聚鸣', '趣时', '辰元', '运舟', '远望角', '远策', '重阳', '钦沐', '青骊',
    '领久', '香橙',
]))

STRATEGY_KEYWORDS = {
    '300指增': ['300指增', '300增强', '沪深300', '300指', '中证300'],
    '500指增': ['500指增', '500增强', '中证500', '500指', '中证500指增'],
    '1000指增': ['1000指增', '1000增强', '中证1000', '1000指', '中证1000指增'],
    '2000指增': ['2000指增', '2000增强', '中证2000', '2000指'],
    'A500指增': ['A500指增', 'A500增强', '中证A500', 'A500指', 'A500'],
    '量化选股': ['量化选股', '量化多头', '量选'],
    '小市值指增': ['小市值', '小市值指增'],
    '红利指增': ['红利指增', '红利'],
    '主观多头': ['主观多头', '主观'],
    '量化CTA': ['量化CTA', 'CTA', 'cta'],
    '债券': ['债券', '可转债', '债基'],
    '宏观': ['宏观', '宏观策略', '全天候'],
    '市场中性': ['市场中性', '中性', '对冲'],
    '量化择时': ['量化择时', '择时'],
    '股票多空': ['股票多空', '多空'],
    '全指指增': ['全指指增', '中证全指', '全指'],
    '套利': ['套利'],
}

# 策略 -> 基准宽基指数（知识库里的指数文档名）
STRATEGY_BENCHMARK = {
    '300指增': '沪深300指数',
    '500指增': '中证500指数',
    '1000指增': '中证1000指数',
    '2000指增': '中证2000指数',
    'A500指增': '中证A500指数',
    '红利指增': '中证红利',
    '小市值指增': '万得小市值',
    '全指指增': '中证全指',
    '量化选股': '万得全A',
}

# 组合配置相关：触发词 + 基准指数别名（用于"vs 沪深300"这类显式指定）
PORTFOLIO_INTENT_KEYWORDS = [
    '组合', '配置', '配比', '篮子', '打包', 'FOF', 'fof',
    '综合收益', '整体收益', '跑赢', '跑输', '超额', '基准',
]
BENCHMARK_ALIASES = {
    '沪深300指数': ['沪深300', '沪深 300'],
    '中证500指数': ['中证500', '中证 500'],
    '中证1000指数': ['中证1000', '中证 1000'],
    '中证2000指数': ['中证2000', '中证 2000'],
    '中证A500指数': ['中证A500', '中证 A500', 'A500'],
    '中证红利': ['中证红利'],
    '中证全指': ['中证全指', '全指'],
    '万得全A': ['万得全A', '万得全 a', '全A'],
    '万得小市值': ['万得小市值', '小市值指数'],
}

# 指数文档在知识库里也有"策略类型"字段，解析代表产品时要排除，避免把指数当产品
INDEX_DOC_NAMES = set(STRATEGY_BENCHMARK.values()) | set(BENCHMARK_ALIASES.keys())

RECALL_K = 50
RERANK_TOP_K = 12
FALLBACK_TOP_K = 15
COMPARISON_MAX_DOCS = 25

# 查询意图关键词
COMPARISON_KEYWORDS = ["对比", "比较", "vs", "VS", "区别", "哪个更好", "哪个强", "哪个好", "哪家", "谁更好", "谁强", "谁好"]
RANKING_KEYWORDS = ["谁表现好", "谁最好", "最好", "排名", "最强", "最突出", "收益最高", "表现最好", "哪个产品好"]
CHART_TRIGGER_KEYWORDS = COMPARISON_KEYWORDS + RANKING_KEYWORDS + [
    '走势', '走势图', '画个图', '画图', '图表', '可视化', '折线图', '柱状图', '曲线图', '画一下',
]

_chinese_font_configured = False


def setup_chinese_font():
    """设置中文字体，解决 matplotlib 中文显示问题"""
    global _chinese_font_configured
    if _chinese_font_configured:
        return

    system = platform.system()
    if system == 'Windows':
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'STZhongsong']
    elif system == 'Darwin':
        font_names = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
    else:
        font_names = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']

    for font_name in font_names:
        try:
            font_path = fm.findfont(font_name)
            if font_path:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 使用中文字体: {font_name}")
                _chinese_font_configured = True
                return
        except Exception:
            continue

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 使用保底字体，可能无法显示中文")
    _chinese_font_configured = True


def load_knowledge_qa(file_path='knowledge.txt'):
    """按 \\n\\nQ: 分割，保持每条完整问答"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    qa_blocks = content.split('\n\nQ: ')
    documents = []
    for block in qa_blocks:
        if not block.strip():
            continue
        if not block.startswith('Q:'):
            block = 'Q: ' + block
        if 'A: ' not in block:
            # 兼容个别文档答案行没有 "A: " 前缀（如指数"近期表现"），补上以便解析
            lines = block.split('\n', 1)
            if len(lines) == 2 and lines[1].strip():
                block = lines[0] + '\nA: ' + lines[1]
        if 'A: ' in block:
            documents.append(Document(page_content=block.strip(), metadata={'source': file_path}))

    print(f"📚 加载了 {len(documents)} 条完整问答")
    return documents


# ==================== 图表数据解析与生成 ====================
METRIC_ALIASES = {
    '近1周': ['近1周'],
    '今年以来': ['今年以来', '今年'],
    '成立以来': ['成立以来'],
    '2025年': ['2025年'],
}

_METRIC_LABELS = {
    '近1周': '近1周',
    '今年以来': '今年以来',
    '成立以来': '成立以来',
    '2025年': '2025年',
}


def extract_manager(product_name: str) -> str:
    """从产品名中识别管理人，识别不到就返回产品名本身。"""
    for mgr in sorted(MANAGER_LIST, key=len, reverse=True):
        if mgr in product_name:
            return mgr
    cleaned = product_name.strip()
    cleaned = re.sub(r'^(A:|[：:，,、。；;·•\s])+', '', cleaned)
    return cleaned


def strategy_match_score(product_name: str, keywords: list) -> int:
    return sum(1 for kw in keywords if kw in product_name)


def _fuzzy_manager_match(question: str) -> Optional[str]:
    """错别字容错：精确匹配不到管理人时，用滑窗字符相似度找最接近的。
    如"执行通达" -> "知行通达"。"""
    from difflib import SequenceMatcher

    best_name, best_ratio = None, 0.0
    for mgr in sorted(MANAGER_LIST, key=len, reverse=True):
        mgr_len = len(mgr)
        if mgr_len < 2 or mgr_len > 8:
            continue
        for i in range(0, len(question) - mgr_len + 1):
            window = question[i:i + mgr_len]
            ratio = SequenceMatcher(None, window, mgr).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = mgr
    return best_name if best_ratio >= 0.72 else None


def _strip_code_blocks(text: str) -> str:
    """剔除回答中的 markdown 代码块（含流式中未闭合的围栏），防止模型输出代码。"""
    text = re.sub(r'```[\s\S]*?```', '', text)   # 完整的围栏代码块
    text = re.sub(r'```[\s\S]*?$', '', text)     # 尾部未闭合的围栏（流式中间态）
    # 剔除无围栏的代码行（matplotlib/python 特征）
    kept = []
    for line in text.splitlines():
        if re.match(r'^\s*(import |from |plt\.|matplotlib|fig, |ax\.|np\.|pd\.|DataFrame|savefig|subplots|show\(|print\()', line):
            continue
        kept.append(line)
    return '\n'.join(kept)


def format_chat_history(chat_history: List[dict], limit: int = 6) -> str:
    """把最近对话整理为提示词文本，供普通回答和流式回答共用。

    只保留最后 ``limit`` 条，避免历史无限变长而挤占知识库上下文。
    """
    if not chat_history:
        return "暂无历史对话"

    formatted = []
    for msg in chat_history[-limit:]:
        role = "用户" if msg["role"] == "user" else "助手"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def _strategy_text_match(text: str, strategy: str) -> bool:
    """判断文本是否属于指定指增策略，避免 A500 被 500 误匹配。"""
    if strategy == '500指增':
        return bool(re.search(r'(?<![0-9A-Za-z])500(?:指增|增强)|中证500', text))
    if strategy == '1000指增':
        return bool(re.search(r'(?<![0-9A-Za-z])1000(?:指增|增强)|中证1000', text))
    if strategy == '300指增':
        return bool(re.search(r'(?<![0-9A-Za-z])300(?:指增|增强)|中证300|沪深300', text))
    if strategy == '2000指增':
        return bool(re.search(r'(?<![0-9A-Za-z])2000(?:指增|增强)|中证2000', text))
    if strategy == 'A500指增':
        return bool(re.search(r'A500', text))
    return any(kw in text for kw in STRATEGY_KEYWORDS.get(strategy, [strategy]))


def pick_target_metric(question: str) -> str:
    """根据问题中的时间词选择图表指标。"""
    for kw, metric in [
        ('近1周', '近1周'), ('近一周', '近1周'), ('本周', '近1周'),
        ('成立以来', '成立以来'),
        ('2025', '2025年'),
        ('今年以来', '今年以来'), ('今年', '今年以来'),
    ]:
        if kw in question:
            return metric
    return '今年以来'


def _metric_value(text: str, metric: str) -> Optional[float]:
    """从文本中提取指定指标的第一个数值（不含 % 符号）。"""
    # 去掉"最大回撤X%"片段，避免被误当成收益（如"今年以来最大回撤：-25.92%"）
    text = re.sub(r'最大回撤[^，。；%]*?[+-]?\d+(?:\.\d+)?%', '', text)
    aliases = METRIC_ALIASES[metric]
    pattern = '(?:' + '|'.join(aliases) + r')[^，。；%]*?([+-]?\d+(?:\.\d+)?)%'
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _clean_product_question(q: str) -> str:
    """把"XX的业绩表现如何？"这类问句还原成产品名。"""
    for suffix in ['的业绩表现怎么样', '的业绩表现如何', '各年度收益是多少', '各期收益分别是多少',
                   '的收益情况如何', '的基本情况是什么', '具体情况', '怎么样', '呢']:
        q = q.split(suffix)[0]
    return q.strip().strip('？?')


def parse_portfolio(question: str):
    """解析组合表达式，如"50%的衍复1000指增 + 30%的幻方500指增 + 20%的远澜CTA"。
    兼容"衍复1000指增50%、幻方500指增30%、远澜CTA20%"这类名称在前的写法。
    返回 [(权重, 名称), ...]，不足 2 条或权重异常时返回 None。
    """
    q = question.replace('，', ',').replace('、', ',').replace('；', ';').replace('＋', '+')
    # 用分隔符切段；没有分隔符时按"空格+数字"切（如"50%衍复 ... 30%幻方 ..."）
    segments = [s.strip() for s in re.split(r'[+,;]|\s+(?=\d)', q) if s.strip()]
    legs = []
    for seg in segments:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', seg)
        if not m:
            continue
        weight = float(m.group(1))
        name = (seg[:m.start()] + seg[m.end():]).strip()
        name = name.strip('的占比权重等买持有：: ')
        if not name:
            continue
        legs.append((weight, name))
    if len(legs) < 2:
        return None
    total = sum(w for w, _ in legs)
    if total <= 0:
        return None
    return legs


_CN_FRACTION = {2: '二分之一', 3: '三分之一', 4: '四分之一', 5: '五分之一'}


def parse_product_list(question: str):
    """把"远澜CTA、衍复A500、幻方1000"这类产品列表切成名称列表。
    分隔符：、，,;；+、连接词“和”及空白。至少 2 个名称才返回，否则 None。

    管理人名称可能本身含“和”（如“平方和”“悦海盈和”），因此先用占位符
    保护完整管理人名称，再把其余“和”作为连接词切分。
    """
    q = question.replace('，', ',').replace('、', ',').replace('；', ';').replace('＋', '+')
    protected_managers = {}
    for i, manager in enumerate(sorted(MANAGER_LIST, key=len, reverse=True)):
        if manager in q:
            placeholder = f'__MANAGER_{i}__'
            q = q.replace(manager, placeholder)
            protected_managers[placeholder] = manager
    q = re.sub(r'和', ',', q)
    names = [s.strip() for s in re.split(r'[,+;]|\s+', q) if s.strip()]
    names = [
        _restore_protected_manager(name, protected_managers)
        for name in names
    ]
    if len(names) < 2:
        return None
    return names


def _restore_protected_manager(name: str, protected_managers: dict) -> str:
    """还原 parse_product_list 中临时替换的管理人名称。"""
    for placeholder, manager in protected_managers.items():
        name = name.replace(placeholder, manager)
    return name


# 折线图横轴：用对客户更有意义的"今年以来"口径，不用"近1年"
SERIES_BUCKETS = ['近1周', '近1月', '近3月', '近6月', '今年以来']


def _extract_series(text: str) -> dict:
    """从业绩模板文本中提取各时间段收益序列。
    兼容新旧两种口径："近1周1.35%" 与 "本周收益：8.54%" 都归入近1周。"""
    # 去掉"最大回撤X%"片段，避免误当成收益
    text = re.sub(r'最大回撤[^，。；%]*?[+-]?\d+(?:\.\d+)?%', '', text)
    series = {}
    for m in re.finditer(
            r'(本周|今年以来|今年|近1周|近1月|近3月|近6月|近1年|近2年|近3年|2025年|2024年|2023年|成立以来)[^，。；%]*?([+-]?\d+(?:\.\d+)?)%',
            text):
        # 指数行情文档使用“本周收益 / 今年收益”，产品业绩文档则使用
        # “近1周 / 今年以来”。图表统一使用 SERIES_BUCKETS 的标准口径。
        label = {
            '本周': '近1周',
            '今年': '今年以来',
        }.get(m.group(1), m.group(1))
        if label not in series:
            try:
                series[label] = float(m.group(2))
            except ValueError:
                pass
    return series


def extract_chart_pairs(context: str, metric: str) -> List[tuple]:
    """从检索上下文中提取 (产品名, 数值, 来源问题, 收益序列) 列表，兼容知识库的真实格式。"""
    pairs: List[tuple] = []

    for block in context.split('\n\n'):
        if 'Q:' not in block or 'A:' not in block:
            continue
        q_part, a_part = block.split('A:', 1)
        q_raw = q_part.replace('Q:', '').strip()
        a_text = a_part.strip()

        # 1) 括号式结构化条目：产品名（管理人，今年X%，成立以来X%，最大回撤X%）
        #    覆盖"策略今年表现 / 各管理人表现对比 / 同策略产品对比"类文档
        for match in re.finditer(r'([^（；\n]+?)\s*（([^）]*)）', a_text):
            name = match.group(1).strip()
            val = _metric_value(match.group(2), metric)
            if val is None:
                # 括号内没有指标（如"产品（管理人），今年以来收益X%"），看括号后面的内容
                rest = a_text[match.end():]
                rest = rest.split('（', 1)[0].split('\n\n', 1)[0].split('Q:', 1)[0]
                val = _metric_value(rest, metric)
            if val is not None:
                for prefix in ['今年收益前三：', '排名前三：', '前三名：', '前三：']:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
                # 去掉策略文档的前缀介绍（如"XX策略共N只产品…排名前三："）
                if '：' in name or ':' in name:
                    name = re.split(r'[：:]', name)[-1]
                if name.strip():
                    pairs.append((name.strip(), val, q_raw, {}))

        # 2) "今年收益最高的产品为XX（X%）" / "成立以来收益最高的产品为XX（X%）"
        if metric == '今年以来':
            for match in re.finditer(
                    r'今年收益最高的产品为\s*([^（；，\n]+?)\s*（([+-]?\d+(?:\.\d+)?)%', a_text):
                pairs.append((match.group(1).strip(), float(match.group(2)), q_raw, {}))
        elif metric == '成立以来':
            for match in re.finditer(
                    r'成立以来收益最高的产品为\s*([^（；，\n]+?)\s*（([+-]?\d+(?:\.\d+)?)%', a_text):
                pairs.append((match.group(1).strip(), float(match.group(2)), q_raw, {}))

        # 3) 单产品业绩模板问答：Q 是产品名，A 以"成立以来收益X%"或"近1周X%"开头
        if a_text.startswith('成立以来收益') or a_text.startswith('近1周'):
            name = _clean_product_question(q_raw)
            if name:
                val = _metric_value(a_text, metric)
                if val is not None:
                    pairs.append((name, val, name, _extract_series(a_text)))

    # 按 (产品名, 数值) 去重，优先保留收益序列更完整的条目
    best = {}
    for name, val, source_q, series in pairs:
        key = (name, val)
        if key not in best or len(series) > len(best[key][3]):
            best[key] = (name, val, source_q, series)
    return list(best.values())


def infer_product_strategy(name: str) -> str:
    """从产品名推断策略（如 500指增），识别不到返回空字符串。"""
    for strategy in STRATEGY_KEYWORDS:
        if _strategy_text_match(name, strategy):
            return strategy
    # 数字缩写容错："幻方1000" -> 1000指增、"衍复500" -> 500指增
    m = re.search(r'(\d{3,4})$', name)
    if m and f"{m.group(1)}指增" in STRATEGY_KEYWORDS:
        return f"{m.group(1)}指增"
    return ''


def _strategy_map_from_context(context: str) -> dict:
    """从上下文的基本情况文档中提取 产品名 -> 策略类型（比看名字推断更准）。"""
    smap = {}
    for block in context.split('\n\n'):
        if 'Q:' not in block or 'A:' not in block:
            continue
        q_part, a_part = block.split('A:', 1)
        q = q_part.replace('Q:', '', 1).strip()
        a = a_part.strip()
        if '基本情况' not in q:
            continue
        m = re.search(r'策略类型：([^；；\n]+)', a)
        if m:
            name = _clean_product_question(q)
            if name:
                smap[name] = m.group(1).strip()
    return smap


def _wrap_product_name(name: str, width: int = 14) -> str:
    if len(name) <= width:
        return name
    return name[:width] + '\n' + name[width:]


TECH_COLORS = ['#22D3EE', '#F472B6', '#FBBF24', '#34D399', '#A78BFA', '#FB923C', '#F87171', '#60A5FA']

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')


def _style_tech_ax(ax):
    """深色科技感坐标轴样式。"""
    ax.set_facecolor('#10172B')
    for spine in ax.spines.values():
        spine.set_color('#243252')
    ax.tick_params(colors='#C9D6EE', labelsize=9)
    ax.grid(axis='y', color='#1E2A47', linestyle='--', linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _fig_to_file_url(fig, tag: str) -> str:
    """保存图表为 PNG 文件，返回 Gradio 文件服务 URL。
    相比 base64 内嵌：聊天消息体积小、浏览器渲染稳定、可直接点开。"""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    # 清理超过 24 小时的旧图，避免目录无限增长
    now = datetime.now().timestamp()
    try:
        for name in os.listdir(CHARTS_DIR):
            p = os.path.join(CHARTS_DIR, name)
            if os.path.isfile(p) and now - os.path.getmtime(p) > 86400:
                os.remove(p)
    except Exception:
        pass
    name = f'{tag}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.png'
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return '/gradio_api/file=' + path.replace('\\', '/')


def _render_bar_chart(labels, values, title, metric_label) -> str:
    """霓虹科技风柱状图。"""
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.patch.set_facecolor('#10172B')
    n = len(labels)
    ax.bar(labels, values, color=TECH_COLORS[:n], width=0.62, alpha=0.18, zorder=2)  # 光晕层
    bars = ax.bar(labels, values, color=TECH_COLORS[:n], width=0.5,
                  edgecolor='white', linewidth=1.1, zorder=3)
    for bar, val in zip(bars, values):
        y_pos, va = (val + 0.35, 'bottom') if val >= 0 else (val - 0.35, 'top')
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos, f'{val:+.2f}%',
                ha='center', va=va, color='#EAF2FF', fontsize=10, fontweight='bold', zorder=4)
    max_val, min_val = max(values), min(values)
    y_margin = max(1, (max_val - min_val) * 0.18)
    ax.set_ylim(min_val - y_margin, max_val + y_margin)
    ax.axhline(0, color='#33415E', linewidth=1.1)
    ax.set_title(title, color='#EAF2FF', fontsize=14, fontweight='bold', pad=16)
    ax.set_ylabel(f'{metric_label}收益 (%)', color='#9FB3D1', fontsize=11)
    ax.tick_params(axis='x', labelsize=8.5)
    _style_tech_ax(ax)
    fig.subplots_adjust(bottom=0.22, top=0.88)
    return _fig_to_file_url(fig, 'bar')


def _render_line_chart(items, title) -> str:
    """霓虹科技风多周期走势折线图。items: [(标签, 收益序列 dict), ...]"""
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.patch.set_facecolor('#10172B')
    x_pos = list(range(len(SERIES_BUCKETS)))
    for i, (label, series) in enumerate(items):
        xs, ys = [], []
        for xi, bucket in enumerate(SERIES_BUCKETS):
            if bucket in series and series[bucket] is not None:
                xs.append(xi)
                ys.append(series[bucket])
        if len(xs) < 2:
            continue
        color = TECH_COLORS[i % len(TECH_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=6, alpha=0.15, zorder=2)  # 光晕
        ax.fill_between(xs, ys, 0, alpha=0.06, color=color, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2.6, marker='o', markersize=7,
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.2, zorder=3, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8, color='#EAF2FF', zorder=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SERIES_BUCKETS, color='#C9D6EE', fontsize=10)
    ax.set_title(title, color='#EAF2FF', fontsize=14, fontweight='bold', pad=16)
    ax.set_ylabel('收益 (%)', color='#9FB3D1', fontsize=11)
    ax.axhline(0, color='#33415E', linewidth=1.0)
    _style_tech_ax(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(facecolor='#151E38', edgecolor='#26355B', labelcolor='#DCE7FB',
                  fontsize=8.5, loc='best', framealpha=0.95)
    fig.subplots_adjust(bottom=0.12, top=0.88)
    return _fig_to_file_url(fig, 'line')


def _render_benchmark_chart(product_name: str, product_series: dict,
                            index_name: str, index_series: dict) -> str:
    """产品 vs 基准指数的双线走势对比图（深色科技风）。"""
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.patch.set_facecolor('#10172B')
    x_pos = list(range(len(SERIES_BUCKETS)))
    p_xs, p_ys, i_xs, i_ys = [], [], [], []
    for xi, b in enumerate(SERIES_BUCKETS):
        if b in product_series:
            p_xs.append(xi)
            p_ys.append(product_series[b])
        if b in index_series:
            i_xs.append(xi)
            i_ys.append(index_series[b])
    c_prod, c_idx = TECH_COLORS[0], TECH_COLORS[2]
    if len(p_xs) >= 2:
        ax.plot(p_xs, p_ys, color=c_prod, linewidth=7, alpha=0.15, zorder=2)
        ax.plot(p_xs, p_ys, color=c_prod, linewidth=3.2, marker='o', markersize=8,
                markerfacecolor=c_prod, markeredgecolor='white', markeredgewidth=1.3,
                zorder=3, label=product_name)
        for x, y in zip(p_xs, p_ys):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8.5, color=c_prod, zorder=5)
    if len(i_xs) >= 2:
        ax.plot(i_xs, i_ys, color=c_idx, linewidth=7, alpha=0.15, zorder=2)
        ax.plot(i_xs, i_ys, color=c_idx, linewidth=3.2, marker='s', markersize=7,
                markerfacecolor=c_idx, markeredgecolor='white', markeredgewidth=1.3,
                zorder=3, linestyle='--', label=index_name)
        for x, y in zip(i_xs, i_ys):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, -14),
                        ha='center', fontsize=8.5, color=c_idx, zorder=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SERIES_BUCKETS, color='#C9D6EE', fontsize=11)
    ax.set_title(f'{product_name} vs {index_name}（近1周→今年以来）',
                 color='#EAF2FF', fontsize=15, fontweight='bold', pad=16)
    ax.set_ylabel('收益 (%)', color='#9FB3D1', fontsize=12)
    ax.axhline(0, color='#33415E', linewidth=1.0)
    _style_tech_ax(ax)
    ax.legend(facecolor='#151E38', edgecolor='#26355B', labelcolor='#DCE7FB',
              fontsize=11, loc='best', framealpha=0.95)
    fig.subplots_adjust(bottom=0.12, top=0.88)
    return _fig_to_file_url(fig, 'bench')


def _render_portfolio_chart(portfolio_label: str, portfolio_series: dict,
                            bench_label: str, bench_series: dict,
                            excess_bucket: str, excess_pp: float) -> str:
    """组合综合收益 vs 基准指数的双线图，右侧标注关键区间的超额。"""
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.patch.set_facecolor('#10172B')
    x_pos = list(range(len(SERIES_BUCKETS)))
    p_xs, p_ys, i_xs, i_ys = [], [], [], []
    for xi, b in enumerate(SERIES_BUCKETS):
        if b in portfolio_series:
            p_xs.append(xi)
            p_ys.append(portfolio_series[b])
        if b in bench_series:
            i_xs.append(xi)
            i_ys.append(bench_series[b])
    c_port, c_bench = TECH_COLORS[0], TECH_COLORS[2]
    if len(p_xs) >= 2:
        ax.plot(p_xs, p_ys, color=c_port, linewidth=7, alpha=0.15, zorder=2)
        ax.plot(p_xs, p_ys, color=c_port, linewidth=3.2, marker='o', markersize=8,
                markerfacecolor=c_port, markeredgecolor='white', markeredgewidth=1.3,
                zorder=3, label=portfolio_label)
        for x, y in zip(p_xs, p_ys):
            ax.annotate(f'{y:+.2f}%', (x, y), textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8.5, color=c_port, zorder=5)
    if len(i_xs) >= 2:
        ax.plot(i_xs, i_ys, color=c_bench, linewidth=7, alpha=0.15, zorder=2)
        ax.plot(i_xs, i_ys, color=c_bench, linewidth=3.2, marker='s', markersize=7,
                markerfacecolor=c_bench, markeredgecolor='white', markeredgewidth=1.3,
                zorder=3, linestyle='--', label=bench_label)
        for x, y in zip(i_xs, i_ys):
            ax.annotate(f'{y:+.2f}%', (x, y), textcoords='offset points', xytext=(0, -14),
                        ha='center', fontsize=8.5, color=c_bench, zorder=5)
    # 关键区间超额标注（默认今年以来）
    if excess_bucket and excess_pp is not None:
        try:
            ex_idx = SERIES_BUCKETS.index(excess_bucket)
        except ValueError:
            ex_idx = len(SERIES_BUCKETS) - 1
        color = '#4ADE80' if excess_pp >= 0 else '#F87171'
        ax.annotate(
            f'{excess_bucket}超额 {excess_pp:+.2f}pp',
            xy=(ex_idx, max(p_ys) if p_ys else 0),
            xytext=(ex_idx + 0.15, max(p_ys) + 3 if p_ys else 4),
            fontsize=12, fontweight='bold', color=color, zorder=6,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#151E38',
                      edgecolor=color, linewidth=1.4, alpha=0.95),
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SERIES_BUCKETS, color='#C9D6EE', fontsize=11)
    ax.set_title('组合综合收益 vs 基准指数（近1周→今年以来）',
                 color='#EAF2FF', fontsize=15, fontweight='bold', pad=16)
    ax.set_ylabel('收益 (%)', color='#9FB3D1', fontsize=12)
    ax.axhline(0, color='#33415E', linewidth=1.0)
    _style_tech_ax(ax)
    ax.legend(facecolor='#151E38', edgecolor='#26355B', labelcolor='#DCE7FB',
              fontsize=10, loc='best', framealpha=0.95)
    fig.subplots_adjust(bottom=0.12, top=0.88)
    return _fig_to_file_url(fig, 'portfolio')


def _render_style_overview(product: dict, indices: list) -> str:
    """风格体检总览图：单只产品（粗实线） vs 所有宽基指数（细虚线），看哪条贴得最近。"""
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.patch.set_facecolor('#10172B')
    x_pos = list(range(len(SERIES_BUCKETS)))

    # 指数线：细虚线，弱化
    for i, (idx_name, idx_series) in enumerate(indices):
        xs, ys = [], []
        for xi, b in enumerate(SERIES_BUCKETS):
            if b in idx_series:
                xs.append(xi)
                ys.append(idx_series[b])
        if len(xs) < 2:
            continue
        color = TECH_COLORS[i % len(TECH_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=1.3, linestyle='--', marker='.',
                markersize=3, alpha=0.85, zorder=2, label=idx_name)

    # 产品线：粗实线，突出
    p_xs, p_ys = [], []
    for xi, b in enumerate(SERIES_BUCKETS):
        if b in product['series']:
            p_xs.append(xi)
            p_ys.append(product['series'][b])
    if len(p_xs) >= 2:
        ax.plot(p_xs, p_ys, color='#22D3EE', linewidth=3.8, marker='o', markersize=8,
                markerfacecolor='#22D3EE', markeredgecolor='white', markeredgewidth=1.4,
                zorder=5, label=product['name'])
        for x, y in zip(p_xs, p_ys):
            ax.annotate(f'{y:+.2f}%', (x, y), textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8.5, color='#22D3EE', fontweight='bold', zorder=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(SERIES_BUCKETS, color='#C9D6EE', fontsize=11)
    ax.set_title(f'{product["name"]} vs 所有宽基指数（近1周→今年以来）',
                 color='#EAF2FF', fontsize=15, fontweight='bold', pad=16)
    ax.set_ylabel('收益 (%)', color='#9FB3D1', fontsize=12)
    ax.axhline(0, color='#33415E', linewidth=1.0)
    _style_tech_ax(ax)
    ax.legend(facecolor='#151E38', edgecolor='#26355B', labelcolor='#DCE7FB',
              fontsize=8, loc='best', framealpha=0.95, ncol=2)
    fig.subplots_adjust(bottom=0.12, top=0.88)
    return _fig_to_file_url(fig, 'style_all')


def generate_chart_from_context(context: str, question: str) -> Optional[str]:
    """根据检索上下文和问题生成对比图，返回 base64 data URI；数据不足时返回 None。"""
    if not any(kw in question for kw in CHART_TRIGGER_KEYWORDS):
        return None

    print("📊 尝试生成图表")
    setup_chinese_font()

    metric = pick_target_metric(question)
    all_matches = extract_chart_pairs(context, metric)
    print(f"🔍 匹配到 {len(all_matches)} 条 {metric} 数据")
    if len(all_matches) < 2:
        print("⚠️ 未找到足够的对比数据")
        return None

    target_strategy = ""
    for strategy, keywords in STRATEGY_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            target_strategy = strategy
            break
    if not target_strategy:
        if '指增' in question:
            match = re.search(r'(\d+)指增', question)
            target_strategy = f"{match.group(1)}指增" if match else "指增"
    print(f"🎯 目标策略: {target_strategy or '（未识别）'}")

    keywords_to_match = STRATEGY_KEYWORDS.get(target_strategy, [target_strategy]) if target_strategy else []
    detected_managers = [mgr for mgr in sorted(MANAGER_LIST, key=len, reverse=True) if mgr in question]
    if not detected_managers:
        fuzzy_mgr = _fuzzy_manager_match(question)
        if fuzzy_mgr:
            detected_managers = [fuzzy_mgr]

    def matches_strategy(name: str, source_q: str) -> bool:
        if not target_strategy:
            return False
        return _strategy_text_match(f"{name} {source_q}", target_strategy)

    filtered_matches = all_matches
    if detected_managers:
        # 只保留与问题中明确提到的管理人相关的条目
        manager_matches = [
            (name, val, src, series) for name, val, src, series in all_matches
            if any(mgr in name for mgr in detected_managers) or extract_manager(name) in detected_managers
        ]
        if keywords_to_match:
            filtered_matches = [
                (name, val, src, series) for name, val, src, series in manager_matches
                if matches_strategy(name, src)
            ]
            print(f"🎯 筛选后匹配到 {len(filtered_matches)} 条 {target_strategy} 产品")
            if len(filtered_matches) < 2:
                strategy_num = re.search(r'(\d+)', target_strategy)
                if strategy_num:
                    num = strategy_num.group(1)
                    filtered_matches = [
                        (name, val, src, series) for name, val, src, series in manager_matches
                        if num in name and any(kw in name for kw in ['指增', '增强', '中证', '沪深', '量化'])
                    ]
                print(f"🎯 模糊匹配后找到 {len(filtered_matches)} 条")
        else:
            filtered_matches = manager_matches
    elif keywords_to_match:
        # 策略排名：按策略关键词过滤（产品名或来源文档），不足 2 条时放宽
        filtered_matches = [
            (name, val, src, series) for name, val, src, series in all_matches
            if matches_strategy(name, src)
        ]
        print(f"🎯 筛选后匹配到 {len(filtered_matches)} 条 {target_strategy} 产品")
        if len(filtered_matches) < 2:
            print("⚠️ 策略匹配不足，使用全部产品数据")
            filtered_matches = all_matches

    # 按策略分组：不同策略的产品不能放在同一张图对比
    # （如股票多空 vs 市场中性，风险收益特征不同，同图对比没有意义）
    strategy_map = _strategy_map_from_context(context)

    def group_strategy_of(product_name: str) -> str:
        if target_strategy:
            return target_strategy
        return strategy_map.get(product_name) or infer_product_strategy(product_name) or ''

    # 多管理人对比按管理人去重；单管理人（如"幻方产品走势"）按产品去重
    dedupe_by_product = len(detected_managers) == 1
    groups = {}
    for product_name, val, _source_q, series in filtered_matches[:24]:
        gk = group_strategy_of(product_name)
        if not gk:
            continue  # 识别不出策略的产品不参与对比图，避免跨策略混比
        key = product_name if dedupe_by_product else extract_manager(product_name)
        score = strategy_match_score(product_name, keywords_to_match) if keywords_to_match else 0
        prev = groups.setdefault(gk, {}).get(key)
        if prev is None or score > prev['score'] or (score == prev['score'] and abs(val) > abs(prev['val'])):
            groups[gk][key] = {'val': val, 'score': score, 'product': product_name, 'series': series}

    usable = {gk: items for gk, items in groups.items() if len(items) >= 2}
    if not usable:
        print("⚠️ 同一策略下可对比实体不足 2 个，不生成对比图")
        return None

    metric_label = _METRIC_LABELS[metric]
    parts = []
    try:
        for gk, items in sorted(usable.items()):
            sorted_items = sorted(items.items(), key=lambda x: x[1]['val'], reverse=True)[:8]
            labels, values = [], []
            for _key, data in sorted_items:
                mgr = extract_manager(data['product'])
                strat = strategy_map.get(data['product']) or infer_product_strategy(data['product'])
                head = f'{mgr} · {strat}' if strat else mgr
                labels.append(f'{head}\n{_wrap_product_name(data["product"])}')
                values.append(data['val'])
            print(f"✅ 策略[{gk}] 成功解析 {len(labels)} 个实体: "
                  f"{list(zip([l.split(chr(10))[0] for l in labels], values))}")

            line_items = []
            for _key, data in sorted_items:
                mgr = extract_manager(data['product'])
                points = sum(1 for b in SERIES_BUCKETS if b in data['series'])
                if points >= 3:
                    line_items.append((f"{mgr} · {data['product']}", data['series']))
                if len(line_items) >= 5:
                    break

            gk_tag = f'{gk} ' if gk else ''
            bar_uri = _render_bar_chart(labels, values, f'{gk_tag}产品收益对比（{metric_label}）', metric_label)
            parts.append(f"**📊 {gk}收益对比 · {metric_label}**\n\n![柱状对比图]({bar_uri})")
            if len(line_items) >= 2:
                line_uri = _render_line_chart(line_items, f'{gk_tag}产品多周期收益走势（近1周→今年以来）')
                parts.append(f"**📈 {gk}多周期走势 · 近1周 → 今年以来**\n\n![走势折线图]({line_uri})")
        print("✅ 图表生成成功")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"⚠️ 生成图表失败：{e}")
        import traceback
        traceback.print_exc()
        return None


class AgentState(TypedDict):
    question: str
    chat_history: List[dict]
    context: str
    answer: str


template = """
# 角色设定
你是一位资深投资达人，在小红书上有大量粉丝，擅长用通俗易懂的语言分享量化投资知识。你的风格是：专业但不学术，有温度但不煽情，像一个懂行的朋友在跟你聊天。

## 核心原则
1. **所有数据必须来自【参考资料】**，严禁编造任何数据、产品名称或管理人信息。如果参考资料中没有相关信息，直接说"这个信息我暂时没有找到"。
1. **错别字容错**：如果用户输入的管理人/产品名称与【参考资料】中的名称疑似同一家（如"执行通达"应为"知行通达"），
   按参考资料中的正确名称回答，开头可温和提示正确名称；**不要因为没有精确匹配就回答"没找到"**。
2. **严禁编造对话历史**：绝对不能说"你之前问过XX""你上次提到XX""还记得你问的XX吗"等暗示用户之前问过某个问题的表述，除非【对话历史】中确实记录了该问题。如果对话历史为空或未提及某话题，就当做用户第一次提及。
2. **合规底线**：
   - 绝对不能出现"保本""保收益""稳赚""零风险"等承诺性表述
   - 不能推荐具体购买时点或给出买入建议
   - 历史收益数据必须标注"历史业绩不代表未来表现"
   - 所有涉及收益的数字，必须如实呈现，不夸大不缩小
3. **数据日期（重要）**：内部知识库全部数据截至 {data_date}（含净值、指数收盘、收益统计）。
   当用户问"最新数据到几号/数据日期/数据到什么时候/数据更新到哪天"等问题时，
   必须直接回答"内部知识库数据截至 {data_date}"，**严禁编造任何其他日期**（如当天日期）。
3. **敏感词替换规则**（严格执行）：
   - "私募" → "管理人"
   - "基金"在非产品名称语境下 → "产品"或"策略"
   - 避免出现"买""卖""申购""赎回"等交易引导词，用"了解""关注"替代

## 回答格式
根据用户问题灵活调整，但核心信息需覆盖以下模块（不需要每次都完整输出，按问题相关性取舍）：

### 1️⃣ 管理人速览
用1-2句话概括这家管理人的核心标签（规模量级、成立年限、策略方向），让人快速建立印象。
包含：管理人名称、成立时间、管理规模区间、产品数量及主要类型。

### 2️⃣ 策略方向
归纳该管理人旗下的主要策略线（如300指增、500指增、1000指增、量化选股、市场中性等），简要说明各策略的特点差异。

### 3️⃣ 所有产品及表现
按策略分类列出所有产品，附上关键业绩数据：
- 近1周收益，这个很重要，因为最近表现可能影响未来决策。
- 今年以来收益，这个也很重要，因为历史表现可能影响未来决策。
- 2025年全年收益（如有）
- 成立以来收益/最大回撤（如有）

⚠️ 每段业绩数据后必须附带：「以上为历史数据，不代表未来表现」

### 4️⃣ 一句话总结
用一句话概括该管理人的特点和适合关注的投资者画像。语气要像朋友推荐，不要像广告文案。

## 对比类问题处理
当用户问"XX和YY对比""哪个更好"等对比问题时：
- 从【参考资料】中提取两者的数据进行客观对比
- 只呈现数据差异，不评判"谁更好"
- 用"从数据来看，XX在...方面表现更突出，YY在...方面更有优势"的句式

## 边界问题处理
- 用户问"能赚多少"/"收益怎么样" → 呈现历史数据 + 附加风险提醒
- 用户问"能不能买"/"现在适合入场吗" → 不给出买卖建议，引导"可以关注后续表现，结合自身情况判断"
- 用户问的问题在参考资料中找不到 → 坦诚说"这块信息我手头没有"，不要编

## 语言风格
- 口语化、有温度，像在跟朋友聊天
- 可以用适量emoji，但不要每句话都加
- 避免"本人""笔者""本机构"等生硬自称
- 避免过度专业术语，必须用时加一句大白话解释
- 句子简短，不要用长难句

## 图表要求
- 如果用户要求"画图""走势图""对比图""图表""可视化"等，**绝对不要生成任何代码、代码块、图片链接或 markdown 图片占位符**，
  只用文字描述数据趋势；系统会在回答末尾自动附加生成好的图表。
- 如果用户要对比图/走势图，但问题里**没有明确提到任何管理人、产品或策略**（比如只问"近一周对比图"），
  不要猜着画，先反问澄清，例如："你想对比哪些产品呀？比如衍复和幻方的500指增，或者某只具体产品？"
- 如果某产品没有对应的基准指数数据（如宏观对冲、量化CTA、主观多头等策略没有指数可对比），
  直接说明"知识库暂无该产品的基准指数对比数据"，**不要写代码、不要生成占位图**。

## 输入信息
【数据截止日期】: {data_date}
【对话历史】: {chat_history}
【用户问题】: {question}
【参考资料】: {context}

请开始回答：

"""


# 内部投研/运营 Copilot 的确定性合规护栏。模型提示词仍保留作为第二层，
# 但对“直接推荐、交易时点、收益承诺”这类高风险问题，不交给模型自由发挥。
_ADVICE_PATTERNS = (
    '能买吗', '能不能买', '值得买吗', '推荐买', '推荐哪个', '买哪个',
    '现在买', '什么时候买', '什么时候入场', '建仓', '加仓', '减仓',
    '卖出', '申购', '赎回', '配置多少', '仓位多少', '适合入场', '适合买',
)
_GUARANTEE_PATTERNS = (
    '保本', '保收益', '保证收益', '稳赚', '一定赚钱', '零风险',
)


def compliance_guardrail(question: str) -> Optional[str]:
    """拦截不可由内部工具直接输出的投资建议与收益承诺请求。"""
    normalized = re.sub(r'\s+', '', question or '')
    if any(token in normalized for token in _GUARANTEE_PATTERNS):
        return (
            '⚠️ **合规提示**：我不能确认保本、保收益或零风险。'
            '本工具仅提供内部资料检索与历史数据核验，请由持牌人员进一步确认。'
        )
    if any(token in normalized for token in _ADVICE_PATTERNS):
        return (
            '⚠️ **合规提示**：我不能提供具体买卖、仓位或时点建议。'
            '我可以协助检索产品历史数据、基准对比和风险信息，'
            '具体决策请由持牌人员结合客户适当性情况进一步确认。'
        )
    return None


def source_footer(context: str, max_items: int = 3) -> str:
    """把本次 RAG 上下文中的问答标题作为可读、可复核的检索依据。"""
    labels = []
    for label in re.findall(r'(?:^|\n)Q:\s*([^\n]+)', context or ''):
        label = re.sub(r'\s+', ' ', label).strip()
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= max_items:
            break
    if not labels:
        return ''
    bullets = '\n'.join(f'- 【来源：{label}】' for label in labels)
    return f'\n\n📚 **检索依据（内部知识库）**\n{bullets}'


class RAGApplication:
    """延迟初始化，避免 import 时加载模型与构建向量库。"""

    def __init__(self):
        self.documents = None
        self.vectorstore = None
        self.base_retriever = None
        self.reranker = None
        self.rerank_available = False
        self.llm = None
        self.graph = None
        self._initialized = False

    def initialize(self, knowledge_path='knowledge.txt'):
        if self._initialized:
            return

        # 直接读取 local_config.py 中的本机密钥，不依赖环境变量。
        api_key = DEEPSEEK_API_KEY.strip()
        # 尽早报出配置问题，避免模型、向量库加载完后才发现没有可用密钥。
        if not api_key:
            raise RuntimeError(
                "未检测到 local_config.py 中的 DEEPSEEK_API_KEY。"
                "请检查本机私密配置文件。"
            )

        import kb_management as kbm

        print("🔄 正在初始化 RAG 系统...")
        self.documents = load_knowledge_qa(knowledge_path)
        self.product_catalog = self._build_product_catalog()
        print(f"📇 产品目录构建完成：{len(self.product_catalog)} 只产品")
        try:
            with open(knowledge_path, encoding='utf-8') as _f:
                self.data_date = kbm.data_date_of(_f.read())
        except Exception:
            self.data_date = None
        print(f"📅 知识库数据截止日期：{self.data_date or '未知'}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder='./cache',
            model_kwargs={'local_files_only': True},
        )

        persist_dir = './chroma_db'
        if kbm.should_rebuild(knowledge_path, persist_dir):
            kbm.snapshot_knowledge(knowledge_path)
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            print("🔨 检测到知识库变更或首次构建，正在重建向量库...")
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            print("📂 知识库未变更，直接加载已有向量库...")
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            # 自愈：向量库文档数与当前知识库解析结果不一致时（如加载器升级），自动重建
            try:
                vec_count = self.vectorstore._collection.count()
                if vec_count != len(self.documents):
                    print(f"⚠️ 向量库文档数({vec_count})与知识库({len(self.documents)})不一致，触发重建...")
                    try:
                        self.vectorstore._client.close()
                    except Exception:
                        pass
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                    self.vectorstore = Chroma.from_documents(
                        documents=self.documents,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                    )
            except Exception as e:
                print(f"ℹ️ 向量库文档数自检跳过：{e}")

        kbm.record_build(knowledge_path, len(self.documents))

        self.base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RECALL_K},
        )

        self._init_reranker()
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            model="deepseek-chat",
            temperature=0.3,
        )
        self.graph = self._build_graph()
        self._initialized = True
        print("✅ RAG 系统初始化完成")

    def _init_reranker(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            print("⚠️ transformers 不可用，使用基础向量检索器")
            return

        try:
            print("🔄 正在加载 Rerank 模型...")
            model_name = "BAAI/bge-reranker-v2-m3"
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            model.eval()
            self.reranker = {'tokenizer': tokenizer, 'model': model, 'torch': torch}
            self.rerank_available = True
            print("✅ Rerank 模型加载成功")
        except Exception as e:
            print(f"⚠️ Rerank 模型加载失败: {e}")
            self.reranker = None
            self.rerank_available = False

    def retrieve_with_rerank(self, query: str, manager_name: str = None) -> Optional[str]:
        """向量召回 + Rerank 精排，适用于无实体、模糊语义查询。"""
        optimized_query = f"{manager_name} 旗下产品 业绩表现" if manager_name else query

        docs = self.base_retriever.invoke(optimized_query)
        if not docs:
            print("⚠️ 基础检索未找到结果")
            return None

        print(f"🔍 向量召回 {len(docs)} 个候选文档 (k={RECALL_K})")

        if self.rerank_available and self.reranker:
            try:
                tokenizer = self.reranker['tokenizer']
                model = self.reranker['model']
                torch = self.reranker['torch']

                pairs = [[optimized_query, doc.page_content] for doc in docs]
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=1024,
                )

                with torch.no_grad():
                    scores = model(**inputs).logits.squeeze(-1).tolist()

                if not isinstance(scores, list):
                    scores = [scores]

                scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                docs = [doc for doc, _ in scored_docs[:RERANK_TOP_K]]
                print(f"✅ Rerank 完成，保留 Top {RERANK_TOP_K} 结果")
            except Exception as e:
                print(f"⚠️ Rerank 执行失败: {e}，降级取 Top {FALLBACK_TOP_K}")
                docs = docs[:FALLBACK_TOP_K]
        else:
            docs = docs[:FALLBACK_TOP_K]
            print(f"✅ 使用基础检索结果，取前 {len(docs)} 条")

        if not docs:
            return None
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_all_for_manager(self, manager_name: str) -> Optional[str]:
        """全量关键字扫描：拉取某管理人所有相关文档。用于有明确管理人实体的查询。"""
        matched = []
        for doc in self.documents:
            if manager_name in doc.page_content:
                matched.append(doc.page_content)
        if not matched:
            return None
        result = "\n\n".join(matched[:COMPARISON_MAX_DOCS])
        print(f"📄 {manager_name}: 关键字匹配 {len(matched)} 条，取前 {min(len(matched), COMPARISON_MAX_DOCS)} 条")
        return result

    def retrieve_by_strategy(self, strategy_name: str) -> Optional[str]:
        """全量扫描 + 策略过滤：优先拉取产品名/策略类型直接命中的文档。用于策略排名/策略对比查询。"""
        keywords = STRATEGY_KEYWORDS.get(strategy_name, [strategy_name])
        matched = []
        for doc in self.documents:
            content = doc.page_content
            q_part = content.split('\nA:', 1)[0]
            score = 0
            if any(kw in q_part for kw in keywords):
                score += 2  # 产品名/问题里直接含策略关键词
            if any(kw in content for kw in keywords):
                score += 1  # 正文（策略类型字段等）含策略关键词
            if score:
                matched.append((score, content))
        if not matched:
            return None
        # 按产品去重：每个产品只留一份，优先"业绩表现"文档（含排名所需数据）
        best_by_name = {}
        for score, content in matched:
            q_part = content.split('\nA:', 1)[0].replace('Q:', '', 1).strip()
            name = _clean_product_question(q_part)
            if not name:
                continue
            prev = best_by_name.get(name)
            if prev is None or score > prev[0] or (score == prev[0] and '业绩表现' in q_part):
                best_by_name[name] = (score, content)
        top = sorted(best_by_name.values(), key=lambda x: x[0], reverse=True)[:100]
        result = "\n\n".join(c for _, c in top)
        print(f"📊 {strategy_name}: 策略过滤匹配 {len(matched)} 条文档 / {len(best_by_name)} 只产品，取 {len(top)} 只")
        return result

    def retrieve_by_product_name(self, question: str) -> Optional[str]:
        """反向精确匹配：从问题里找知识库中的产品名/主题词。
        解决向量检索对精确产品名召回不稳（如"久期宏观对冲1号"被漏掉）的问题。"""
        q_clean = re.sub(r'[？?。！!，,、\s]', '', question)
        matched = []
        for doc in self.documents:
            q_part = doc.page_content.split('\nA:', 1)[0].replace('Q:', '', 1).strip()
            name = _clean_product_question(q_part)
            name_clean = re.sub(r'[？?。！!，,、\s]', '', name)
            if len(name_clean) >= 4 and name_clean in q_clean:
                matched.append(doc.page_content)
        if not matched:
            return None
        result = "\n\n".join(matched[:COMPARISON_MAX_DOCS])
        print(f"🔍 [产品名精确匹配] 命中 {len(matched)} 条，取前 {min(len(matched), COMPARISON_MAX_DOCS)} 条")
        return result

    def find_index_series(self, index_name: str) -> Optional[dict]:
        """在知识库里查找宽基指数文档的近期收益序列（近1周→今年以来）。
        兼容新旧命名："中证1000指数" / "中证1000"、"业绩表现" / "近期表现"。"""
        variants = {index_name}
        if index_name.endswith('指数'):
            variants.add(index_name[:-2])
        else:
            variants.add(index_name + '指数')
        for doc in self.documents:
            parts = doc.page_content.split('\nA:', 1)
            if len(parts) != 2:
                continue
            q_part = parts[0].replace('Q:', '', 1).strip()
            if not any(q_part == v or q_part.startswith(v + '的') for v in variants):
                continue
            series = _extract_series(parts[1])
            if sum(1 for b in SERIES_BUCKETS if b in series) >= 3:
                return series
        return None

    def generate_benchmark_chart(self, context: str, question: str) -> Optional[str]:
        """单管理人/单产品查询时，生成"产品 vs 基准指数"的走势对比图。"""
        if not context or len(context.strip()) <= 50:
            return None

        strategy_map = _strategy_map_from_context(context)

        # 问题里指定的策略（如"幻方500指增"），用于筛选产品
        q_strategy = ''
        for strategy, keywords in STRATEGY_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                q_strategy = strategy
                break

        products = []
        for block in context.split('\n\n'):
            if 'Q:' not in block or 'A:' not in block:
                continue
            q_part, a_part = block.split('A:', 1)
            a = a_part.strip()
            if not (a.startswith('成立以来收益') or a.startswith('近1周')):
                continue
            name = _clean_product_question(q_part.replace('Q:', '', 1).strip())
            if not name:
                continue
            strat = strategy_map.get(name) or infer_product_strategy(name)
            if not strat or strat not in STRATEGY_BENCHMARK:
                continue
            if q_strategy and strat != q_strategy:
                continue
            series = _extract_series(a)
            if sum(1 for b in SERIES_BUCKETS if b in series) < 3:
                continue
            products.append({'name': name, 'series': series, 'strategy': strat})
        if not products:
            return None

        seen, uniq = set(), []
        for p in products:
            if p['name'] not in seen:
                seen.add(p['name'])
                uniq.append(p)
        uniq = uniq[:3]

        setup_chinese_font()
        parts = []
        for p in uniq:
            idx_name = STRATEGY_BENCHMARK[p['strategy']]
            idx_series = self.find_index_series(idx_name)
            if not idx_series:
                continue
            uri = _render_benchmark_chart(p['name'], p['series'], idx_name, idx_series)
            parts.append(f"**📉 {p['name']} vs {idx_name}**\n\n![产品-基准对比图]({uri})")
        if not parts:
            return None
        print(f"✅ 生成 {len(parts)} 张产品-基准对比图")
        return "\n\n".join(parts)

    # ==================== 组合配置：多产品加权收益 vs 基准 ====================

    def _build_product_catalog(self) -> dict:
        """从知识库构建 产品名 -> {manager, strategy, since, series} 目录。"""
        catalog = {}
        for doc in self.documents:
            parts = doc.page_content.split('\nA:', 1)
            if len(parts) != 2:
                continue
            q_part = parts[0].replace('Q:', '', 1).strip()
            a_part = parts[1].strip()
            name = _clean_product_question(q_part)
            if not name:
                continue
            meta = catalog.setdefault(name, {})
            if '基本情况' in q_part:
                m_mgr = re.search(r'管理人：([^；;\n]+)', a_part)
                m_strat = re.search(r'策略类型：([^；;\n]+)', a_part)
                m_since = re.search(r'成立时间：([^；;\n]+)', a_part)
                if m_mgr:
                    meta['manager'] = m_mgr.group(1).strip()
                if m_strat:
                    meta['strategy'] = m_strat.group(1).strip()
                if m_since:
                    meta['since'] = m_since.group(1).strip()
            if '业绩表现' in q_part:
                series = _extract_series(a_part)
                if series:
                    meta['series'] = series
        # 提及度：同策略多产品时，知识库讨论更多的产品更可能是代表产品
        for name in catalog:
            catalog[name]['mentions'] = sum(
                1 for doc in self.documents if name in doc.page_content
            )
        return catalog

    def _pick_best_product(self, names: list) -> str:
        """多只同策略产品里挑数据最全、知识库提及最多、成立最早的一只作为代表。"""
        def sort_key(name: str):
            meta = self.product_catalog.get(name, {})
            buckets = sum(1 for b in SERIES_BUCKETS if b in meta.get('series', {}))
            since = meta.get('since', '')
            since_key = since if re.match(r'\d{4}-\d{2}-\d{2}', since) else '9999'
            return (-buckets, -meta.get('mentions', 0), since_key, name)
        return sorted(names, key=sort_key)[0]

    def resolve_portfolio_leg(self, name: str) -> Optional[dict]:
        """把组合腿解析成知识库里的具体产品。
        支持三种粒度：
        - 精确产品名：衍复优享中证1000指数增强一号（auto_mapped=None）
        - 管理人+策略："衍复1000指增"（auto_mapped='manager'）
        - 纯策略线："1000指增 / CTA"（auto_mapped='strategy'，自动选代表产品）
        """
        catalog = self.product_catalog
        if name in catalog and catalog[name].get('series'):
            return {'name': name, **catalog[name]}

        mgr = extract_manager(name)
        if mgr not in MANAGER_LIST:
            # extract_manager 没命中管理人时会原样返回名称，视为未指定管理人
            mgr = None
        strat = infer_product_strategy(name)
        if strat:
            # 策略线 / 管理人+策略：优先按策略解析，避免被产品名子串劫持
            candidates = [
                n for n, m in catalog.items()
                if m.get('strategy') == strat and m.get('series') and n not in INDEX_DOC_NAMES
            ]
            if mgr:
                mgr_candidates = [n for n in candidates if catalog[n].get('manager') == mgr]
                if mgr_candidates:
                    n = self._pick_best_product(mgr_candidates)
                    return {'name': n, 'auto_mapped': 'manager', **catalog[n]}
                # 该管理人无此策略产品（如"衍复A500"）：退回同类策略代表产品并注明
                if candidates:
                    n = self._pick_best_product(candidates)
                    return {
                        'name': n,
                        'auto_mapped': 'strategy',
                        'unmatched_manager': mgr,
                        **catalog[n],
                    }
            if candidates:
                n = self._pick_best_product(candidates)
                return {'name': n, 'auto_mapped': 'manager' if mgr else 'strategy', **catalog[n]}

        # 部分产品名（如 "远澜红枫" -> 远澜红枫1号）
        sub_matches = [n for n, m in catalog.items() if name in n and m.get('series')]
        if len(sub_matches) == 1:
            n = sub_matches[0]
            return {'name': n, **catalog[n]}
        if len(sub_matches) > 1:
            n = self._pick_best_product(sub_matches)
            return {'name': n, **catalog[n]}
        return None

    def portfolio_benchmark_options(self, question: str, resolved: list) -> dict:
        """构建多维度基准：各底层指数（按仓位降序）+ 合成基准 + CTA无基准视图。
        默认基准 = 提问显式指定的指数 > 权重最大的底层指数（主要贝塔来源）。
        """
        index_weights = {}   # 底层指数 -> 对应仓位权重
        cta_legs = []        # 无宽基映射的腿（CTA/宏观等）
        for r in resolved:
            idx = STRATEGY_BENCHMARK.get(r['strategy'])
            if idx:
                index_weights[idx] = index_weights.get(idx, 0.0) + r['weight_norm']
            else:
                cta_legs.append(r)

        idx_series = {}
        for idx in index_weights:
            s = self.find_index_series(idx)
            if s:
                idx_series[idx] = s

        options, series_map = [], {}
        # 1) 各底层指数：仓位越大越靠前（主要贝塔优先）
        for idx_name, w in sorted(index_weights.items(), key=lambda x: -x[1]):
            if idx_name in idx_series:
                key = f'idx:{idx_name}'
                options.append((f'{idx_name}（对应{w:.0f}%仓位）', key))
                series_map[key] = idx_series[idx_name]

        # 2) 合成基准：按指数映射腿的仓位加权
        if idx_series:
            key = 'composite'
            total = sum(index_weights[i] for i in idx_series)
            comp = {}
            for b in SERIES_BUCKETS:
                vals = [(index_weights[i] / total, idx_series[i].get(b)) for i in idx_series]
                vals = [(w, v) for w, v in vals if v is not None]
                if vals:
                    comp[b] = sum(w * v for w, v in vals)
            label = ' + '.join(f'{index_weights[i] / total:.0%}{i}' for i in idx_series)
            options.append((f'合成基准（{label}）', key))
            series_map[key] = comp

        # 3) CTA/宏观等无基准腿：单独视图，展示绝对收益
        if cta_legs:
            key = 'cta'
            cta_series = {}
            for b in SERIES_BUCKETS:
                vals = [(r['weight_norm'], r['series'].get(b)) for r in cta_legs]
                vals = [(w, v) for w, v in vals if v is not None]
                if vals:
                    wsum = sum(w for w, _ in vals)
                    cta_series[b] = sum(w * v for w, v in vals) / wsum
            strat_names = '/'.join(dict.fromkeys(r['strategy'] for r in cta_legs))
            options.append((f'{strat_names}（无基准）', key))
            series_map[key] = cta_series

        # 4) 常用宽基指数：让用户自由选择对比视角（不限于组合内出现的指数）
        for idx_name in BENCHMARK_ALIASES:
            key = f'idx:{idx_name}'
            if key in series_map:
                continue
            s = self.find_index_series(idx_name)
            if s:
                options.append((idx_name, key))
                series_map[key] = s

        # 默认基准：显式指定 > 仓位最大的底层指数 > 合成基准 > CTA
        default = None
        # 先把产品名从问题里去掉，避免"衍复A500"里的 A500 被当成显式基准
        q_sans_legs = question
        for r in resolved:
            q_sans_legs = q_sans_legs.replace(r['raw_name'], '')
        for idx_name, aliases in BENCHMARK_ALIASES.items():
            if any(a in q_sans_legs for a in aliases):
                k = f'idx:{idx_name}'
                if k in series_map:
                    default = k
                break
        if default is None and idx_series:
            top_idx = max(idx_series, key=index_weights.get)
            default = f'idx:{top_idx}'
        if default is None:
            default = 'composite' if 'composite' in series_map else ('cta' if 'cta' in series_map else None)
        return {'options': options, 'default': default, 'series_map': series_map}

    def build_portfolio(self, question: str, benchmark_key: Optional[str] = None,
                        require_intent: bool = True) -> Optional[dict]:
        """解析组合问题并计算加权收益 + 可选基准。
        require_intent=True 时要求问题包含组合/基准/跑赢等意图词（聊天场景防误触发）；
        组合分析器面板场景传 False，用户已明确表达了意图。"""
        if not getattr(self, 'product_catalog', None):
            self.initialize()
        parsed = parse_portfolio(question)
        if require_intent and not any(kw in question for kw in PORTFOLIO_INTENT_KEYWORDS):
            return None
        if parsed is None:
            # 未带比例时，把产品列表按等权处理（客户场景常见："远澜CTA、衍复A500、幻方1000"）
            names = parse_product_list(question)
            if not names:
                return None
            parsed = [(100.0 / len(names), n) for n in names]

        resolved, notes = [], []
        for weight, raw_name in parsed:
            leg = self.resolve_portfolio_leg(raw_name)
            if not leg:
                notes.append(f"⚠️ 知识库未找到「{raw_name}」的业绩数据，已跳过")
                continue
            leg['weight'] = weight
            leg['raw_name'] = raw_name
            if leg.get('unmatched_manager'):
                notes.append(
                    f"⚠️ 「{raw_name}」：{leg['unmatched_manager']}暂无{leg['strategy']}产品数据，"
                    f"改用同类代表产品 {leg['name']} 估算"
                )
            resolved.append(leg)
        if len(resolved) < 2:
            return None

        total_w = sum(r['weight'] for r in resolved)
        if abs(total_w - 100) > 1:
            notes.append(f"ℹ️ 权重合计 {total_w:.0f}%，已按比例归一化到 100%")
        if all(abs(r['weight'] - 100.0 / len(resolved)) < 0.01 for r in resolved):
            frac = _CN_FRACTION.get(len(resolved), f'1/{len(resolved)}')
            notes.append(f"ℹ️ 未提供比例，按等权估算（{len(resolved)}只产品各占约{frac}）")
        for r in resolved:
            r['weight_norm'] = r['weight'] / total_w * 100

        # 组合综合收益：按期初权重对各区间收益线性加权
        portfolio_series = {}
        for b in SERIES_BUCKETS:
            vals = [(r['weight_norm'], r['series'].get(b)) for r in resolved]
            vals = [(w, v) for w, v in vals if v is not None]
            if vals:
                portfolio_series[b] = sum(w * v for w, v in vals) / 100.0

        bench_opts = self.portfolio_benchmark_options(question, resolved)
        if not bench_opts['options']:
            notes.append("⚠️ 未找到可用的基准指数数据，无法对比")
            return None
        avail_keys = [k for _, k in bench_opts['options']]
        bench_key = benchmark_key if benchmark_key in avail_keys else bench_opts['default']
        bench_label = dict((k, label) for label, k in bench_opts['options'])[bench_key]
        bench_series = bench_opts['series_map'][bench_key]
        if bench_key == 'composite':
            notes.append(
                "ℹ️ 合成基准 = 有宽基映射各腿的仓位加权指数；"
                "CTA/宏观等绝对收益策略无对应宽基，不参与合成"
            )
        if bench_key == 'cta':
            notes.append(
                "ℹ️ CTA/宏观等绝对收益策略没有单一宽基基准，无法回答“跑赢/跑输市场”，"
                "此处仅展示其绝对收益供参考"
            )

        return {
            'legs': resolved,
            'portfolio_series': portfolio_series,
            'bench_key': bench_key,
            'bench_label': bench_label,
            'bench_series': bench_series,
            'bench_options': bench_opts['options'],
            'default_bench_key': bench_opts['default'],
            'notes': notes,
        }

    def _portfolio_return(self, legs: list, bucket: str) -> Optional[float]:
        """按当前权重计算组合在某区间的加权收益。"""
        vals = [(r['weight_norm'], r['series'].get(bucket)) for r in legs]
        vals = [(w, v) for w, v in vals if v is not None]
        if not vals:
            return None
        return sum(w * v for w, v in vals) / 100.0

    def _weight_sensitivity(self, legs: list, bucket: str, bench_value: float, step: float = 10.0):
        """粗略敏感性：逐腿权重 ±step 个百分点（其余按比例缩放），
        返回 (区间低, 区间高, 结论是否可能反转)。"""
        base_ret = self._portfolio_return(legs, bucket)
        if base_ret is None or bench_value is None:
            return None
        returns = []
        n = len(legs)
        for i, leg in enumerate(legs):
            for delta in (step, -step):
                w_i = min(100.0, max(0.0, leg['weight_norm'] + delta))
                other_total = 100.0 - w_i
                other_w = sum(r['weight_norm'] for j, r in enumerate(legs) if j != i)
                if other_w <= 0:
                    continue
                vals = []
                for j, r in enumerate(legs):
                    w = w_i if j == i else r['weight_norm'] / other_w * other_total
                    v = r['series'].get(bucket)
                    if v is None:
                        continue
                    vals.append((w, v))
                if vals:
                    returns.append(sum(w * v for w, v in vals) / 100.0)
        if not returns:
            return None
        lo, hi = min(returns), max(returns)
        flips = any((r - bench_value) * (base_ret - bench_value) < 0 for r in returns)
        return lo, hi, flips

    def generate_portfolio_analysis(self, question: str, benchmark_key: Optional[str] = None,
                                    require_intent: bool = True,
                                    client_mode: Optional[bool] = None) -> Optional[str]:
        """生成"组合 vs 基准"的完整分析：结论 + 数据表 + 双线图 + 分产品贡献。
        client_mode=True 时走客户版口径：不出现产品名/权重百分比，用白话表述并附带免责说明。
        client_mode=None 时自动判断：带比例输入走顾问口径，纯产品列表（等权）走客户口径。"""
        try:
            data = self.build_portfolio(question, benchmark_key, require_intent=require_intent)
            if not data:
                return None

            setup_chinese_font()
            client = client_mode if client_mode is not None else (parse_portfolio(question) is None)
            legs, p_series, b_series = data['legs'], data['portfolio_series'], data['bench_series']
            bench_label = data['bench_label']
            bench_label_display = bench_label.split('（')[0] if client else bench_label

            def leg_display(r: dict) -> str:
                if client:
                    cheng = max(1, round(r['weight_norm'] / 10))
                    return f"{r['strategy']}约{cheng}成"
                mark = {
                    'strategy': '·代表产品',
                    'manager': '·管理人自动匹配',
                }.get(r.get('auto_mapped'), '')
                return f"{r['weight_norm']:.0f}% {r['name']}（{r['strategy']}{mark}）"

            auto_legs = [r for r in legs if r.get('auto_mapped')]
            if auto_legs:
                if client:
                    data['notes'].append(
                        "ℹ️ 按行业代表产品估算：" + "、".join(
                            f"{r['strategy']}→{r['name']}" for r in auto_legs
                        ) + "；您的实际持仓可能与此不同"
                    )
                else:
                    data['notes'].append(
                        "ℹ️ 自动映射：" + "、".join(
                            f"{r['raw_name']} → {r['name']}"
                            + ("（代表产品）" if r['auto_mapped'] == 'strategy' else "（管理人自动匹配）")
                            for r in auto_legs
                        ) + "；需要精确口径时请直接输入具体产品名"
                    )

            # CTA 无基准视图：只展示绝对收益，不做跑赢/跑输结论
            if data['bench_key'] == 'cta':
                cta_items = [
                    (leg_display(r), r['series'])
                    for r in legs if r['strategy'] not in STRATEGY_BENCHMARK
                ]
                uri = _render_line_chart(cta_items, 'CTA腿绝对收益（无宽基基准可比）')
                rows = []
                for b in SERIES_BUCKETS:
                    if b in b_series:
                        rows.append(f"| {b} | {b_series[b]:+.2f}% |")
                table = '\n'.join(rows)
                md = (
                    f"**🧭 CTA（无基准）视图**\n\n"
                    f"CTA/宏观等绝对收益策略没有单一宽基基准，无法直接回答“跑赢/跑输市场”。"
                    f"它的价值在于危机保护与分散，单独看绝对收益更合理。\n\n"
                    f"**CTA腿加权收益**\n\n| 区间 | 收益 |\n|---|---|\n{table}\n\n"
                    f"![CTA绝对收益]({uri})"
                )
                if data['notes']:
                    md += "\n\n**口径说明**\n" + '\n'.join(f'- {n}' for n in data['notes'])
                print("✅ CTA 无基准视图生成成功")
                return md

            # 结论区间：优先今年以来，缺失则取最近可对比区间
            key_bucket, key_excess = None, None
            for b in reversed(SERIES_BUCKETS):
                if b in p_series and b in b_series:
                    key_bucket, key_excess = b, p_series[b] - b_series[b]
                    break
            if key_bucket is None:
                return None

            if client:
                verdict = (
                    f"{key_bucket}，这类配置约 {p_series[key_bucket]:+.1f}%，"
                    f"同期{bench_label_display}约 {b_series[key_bucket]:+.1f}%，"
                    f"{'跑赢' if key_excess >= 0 else '跑输'}约 {abs(key_excess):.1f} 个百分点"
                )
            else:
                verdict = (
                    f"{key_bucket}组合 {'跑赢' if key_excess >= 0 else '跑输'} 基准 "
                    f"{abs(key_excess):.2f} 个百分点"
                )
            emoji = '✅' if key_excess >= 0 else '⚠️'

            leg_names = ' + '.join(leg_display(r) for r in legs)
            if client:
                port_label = '我的组合'
                ws = [r['weight_norm'] for r in legs]
                equal = max(ws) - min(ws) < 0.5
                if equal:
                    frac = _CN_FRACTION.get(len(legs), f'1/{len(legs)}')
                    assumed_line = f"假设{len(legs)}只产品各占约{frac}（等权）"
                else:
                    assumed_line = "按这类配置的常见比例假设：" + '、'.join(
                        f"{r['strategy']}约{max(1, round(r['weight_norm'] / 10))}成"
                        for r in legs
                    )
            else:
                port_label = ' + '.join(f"{r['weight_norm']:.0f}%{r['raw_name']}" for r in legs)

            rows = []
            for b in SERIES_BUCKETS:
                pv, bv = p_series.get(b), b_series.get(b)
                if pv is None or bv is None:
                    continue
                rows.append(f"| {b} | {pv:+.2f}% | {bv:+.2f}% | {pv - bv:+.2f}pp |")
            table = '\n'.join(rows)

            sens = self._weight_sensitivity(legs, key_bucket, b_series[key_bucket])
            sens_text = ''
            if sens:
                lo, hi, flips = sens
                if client:
                    sens_text = (
                        f"**比例记不准也没关系**：即使各类配置的比例偏离一些（约±10pp），"
                        f"这类配置的{key_bucket}收益大致在 [{lo:+.1f}%, {hi:+.1f}%] 之间，"
                        f"{'结论方向可能变化，可以再聊聊您实际的大致比例' if flips else '跑赢/跑输结论不变'}。\n\n"
                    )
                else:
                    sens_text = (
                        f"**权重敏感性（{key_bucket}）**：各腿权重±10pp 扰动下，组合收益区间 "
                        f"[{lo:+.2f}%, {hi:+.2f}%]；"
                        f"{'结论方向可能反转，建议与客户确认实际比例' if flips else '跑赢/跑输结论稳定，对比例不敏感'}。\n\n"
                    )

            # 分产品贡献（按结论区间）
            contrib = []
            for r in legs:
                v = r['series'].get(key_bucket)
                if v is None:
                    continue
                contrib.append((r, r['weight_norm'] * v / 100.0, v))
            contrib.sort(key=lambda x: x[1], reverse=True)
            if client:
                contrib_lines = '\n'.join(
                    f"- {r['strategy']}（约{max(1, round(r['weight_norm'] / 10))}成，"
                    f"{key_bucket}{v:+.2f}%）→ 贡献 {c:+.2f}pp"
                    for r, c, v in contrib
                )
            else:
                contrib_lines = '\n'.join(
                    f"- {r['name']}（权重 {r['weight_norm']:.0f}%，{key_bucket}{v:+.2f}%）→ 贡献 {c:+.2f}pp"
                    for r, c, v in contrib
                )

            uri = _render_portfolio_chart(
                port_label, p_series, bench_label_display, b_series, key_bucket, key_excess
            )
            note_text = '\n'.join(f'- {n}' for n in data['notes']) if data['notes'] else ''
            if client:
                raw_names = '、'.join(r['raw_name'] for r in legs)
                header = f"**您的组合**：{raw_names}\n\n{assumed_line}。\n\n"
                md = (
                    f"{header}"
                    f"**{emoji} 结论：{verdict}**\n\n"
                    f"| 区间 | 这类配置 | {bench_label_display} | 超额 |\n"
                    f"|---|---|---|---|\n{table}\n\n"
                    f"{sens_text}"
                    f"**哪部分贡献最大（{key_bucket}）**\n{contrib_lines}\n\n"
                    f"![配置 vs 基准]({uri})\n\n"
                    f"*以上为历史数据，不代表未来表现，不构成投资建议；"
                    f"具体持仓请以您的实际配置为准。*"
                )
            else:
                md = (
                    f"**{emoji} 结论：{verdict}**\n\n"
                    f"**组合构成**：{leg_names}\n\n"
                    f"**对比基准**：{bench_label}\n\n"
                    f"| 区间 | 组合 | 基准 | 超额 |\n|---|---|---|---|\n{table}\n\n"
                    f"{sens_text}"
                    f"**分产品贡献（{key_bucket}）**\n{contrib_lines}\n\n"
                    f"![组合 vs 基准]({uri})"
                )
            if note_text:
                md += f"\n\n**口径说明**\n{note_text}"
            others = [label for label, k in data['bench_options'] if k != data['bench_key']]
            if others:
                if client:
                    md += f"\n\n**换个角度对比**：{' | '.join(o.split('（')[0] for o in others)}"
                else:
                    md += f"\n\n**其他对照基准**：{' | '.join(others)}（可在上方「组合分析器」中切换）"
            print(f"✅ 组合分析生成成功（{len(legs)} 只产品，基准：{bench_label}）")
            return md
        except Exception as e:
            print(f"⚠️ 组合分析生成失败：{e}")
            import traceback
            traceback.print_exc()
            return None

    # ==================== 风格体检：产品 vs 所有宽基指数 ====================

    def list_manager_products(self, manager_query: str, limit: int = 6) -> List[dict]:
        """某管理人旗下有业绩序列的产品（按提及度排序）；输入产品名时返回该产品本身。"""
        catalog = self.product_catalog
        query = manager_query.strip()
        if query in catalog and catalog[query].get('series'):
            return [{'name': query, **catalog[query]}]
        mgr = query if query in MANAGER_LIST else None
        if mgr is None:
            fuzzy = _fuzzy_manager_match(query)
            mgr = fuzzy or extract_manager(query)
        if mgr not in MANAGER_LIST:
            return []
        cands = [
            {'name': n, **m}
            for n, m in catalog.items()
            if m.get('manager') == mgr and m.get('series') and n not in INDEX_DOC_NAMES
        ]
        cands.sort(key=lambda x: (-x.get('mentions', 0), x.get('since', ''), x['name']))
        return cands[:limit]

    def available_indices(self) -> List[tuple]:
        """知识库里有近期收益数据的宽基指数列表 [(指数名, 收益序列)]。"""
        out = []
        for idx_name in BENCHMARK_ALIASES:
            s = self.find_index_series(idx_name)
            if s:
                out.append((idx_name, s))
        return out

    def style_similarity(self, product_series: dict, index_series: dict) -> Optional[dict]:
        """用各区间收益的平均绝对偏差衡量产品与指数的风格接近度（越小越像）。"""
        diffs = []
        for b in SERIES_BUCKETS:
            if b in product_series and b in index_series:
                diffs.append(abs(product_series[b] - index_series[b]))
        if not diffs:
            return None
        return {'mad': sum(diffs) / len(diffs), 'n': len(diffs)}

    def generate_style_analysis(self, manager_query: str,
                                product_key: Optional[str] = None,
                                index_key: Optional[str] = None):
        """风格体检：单只产品 vs 所有宽基指数（总览图 + 接近度表），
        再细分到选定的单个指数看两线详细对比。
        返回 (markdown, 产品选项, 默认产品key, 指数选项, 默认指数key)。"""
        try:
            setup_chinese_font()
            products = self.list_manager_products(manager_query)
            if not products:
                return None, [], None, [], None
            indices = self.available_indices()
            if not indices:
                return None, [], None, [], None

            prod_options = [(p['name'], p['name']) for p in products]
            prod_keys = [k for _, k in prod_options]
            if product_key not in prod_keys:
                product_key = prod_keys[0]
            product = next(p for p in products if p['name'] == product_key)

            # 该产品与每个指数的接近度（平均绝对偏差，越小越像）
            sims = []
            for idx_name, idx_series in indices:
                sim = self.style_similarity(product['series'], idx_series)
                if sim:
                    sims.append((idx_name, sim['mad'], idx_series))
            sims.sort(key=lambda x: x[1])
            if not sims:
                return None, [], None, [], None
            closest_idx = sims[0][0]

            idx_options = [(idx_name, f'idx:{idx_name}') for idx_name, _ in indices]
            idx_keys = [k for _, k in idx_options]
            if index_key not in idx_keys:
                index_key = f'idx:{closest_idx}'

            uri_overview = _render_style_overview(product, indices)
            rows = []
            for idx_name, mad, idx_series in sims:
                ytd_p = product['series'].get('今年以来')
                ytd_i = idx_series.get('今年以来')
                excess = (ytd_p - ytd_i) if (ytd_p is not None and ytd_i is not None) else None
                excess_s = f"{excess:+.2f}pp" if excess is not None else '—'
                rows.append(f"| {idx_name} | {mad:.2f}pp | {excess_s} |")
            table = '\n'.join(rows)

            md_parts = [
                f"**🔎 风格体检：{product['name']} vs 所有宽基指数**",
                f"![产品 vs 所有指数]({uri_overview})",
                f"**风格归属**：{product['name']} 最接近 **{closest_idx}**"
                f"（各区间收益平均偏差 {sims[0][1]:.2f}pp）。",
                f"**与各指数的接近程度（平均偏差越小越像）**\n\n"
                f"| 指数 | 平均偏差 | 今年以来超额 |\n|---|---|---|\n{table}",
                f"**怎么看**：如果一只“1000指增”产品最接近的指数是中证500，"
                f"说明它近期实际风格可能漂向了500，需要留意。"
                f"风格归属仅供参考，不代表未来表现。",
            ]

            # 细分：选定的单个指数 → 两线详细对比
            detail_idx_name = dict((k, label) for label, k in idx_options)[index_key]
            detail_idx_series = dict(indices)[detail_idx_name]
            uri_detail = _render_benchmark_chart(
                product['name'], product['series'], detail_idx_name, detail_idx_series
            )
            detail_rows = []
            for b in SERIES_BUCKETS:
                pv = product['series'].get(b)
                iv = detail_idx_series.get(b)
                if pv is None or iv is None:
                    continue
                detail_rows.append(f"| {b} | {pv:+.2f}% | {iv:+.2f}% | {pv - iv:+.2f}pp |")
            detail_table = '\n'.join(detail_rows)
            md_parts.append(
                f"**细分对比：{product['name']} vs {detail_idx_name}**\n\n"
                f"![细分对比]({uri_detail})\n\n"
                f"| 区间 | {product['name']} | {detail_idx_name} | 超额 |\n"
                f"|---|---|---|---|\n{detail_table}"
            )

            print(f"✅ 风格体检生成成功（{product['name']}，细分：{detail_idx_name}）")
            return '\n\n'.join(md_parts), prod_options, product_key, idx_options, index_key
        except Exception as e:
            print(f"⚠️ 风格体检生成失败：{e}")
            import traceback
            traceback.print_exc()
            return None, [], None, [], None

    def _detect_query_intent(self, question: str):
        """识别问题中的管理人、策略和问法类型。

        这是检索路由的唯一入口；图工作流与流式回答共用它，确保同一句话
        不会因入口不同而走到两套不同的资料检索逻辑。
        """
        sorted_managers = sorted(MANAGER_LIST, key=len, reverse=True)
        detected_managers = list(dict.fromkeys(
            mgr for mgr in sorted_managers if mgr in question
        ))
        if not detected_managers:
            fuzzy_mgr = _fuzzy_manager_match(question)
            if fuzzy_mgr:
                print(f"🔍 [模糊匹配] 管理人：{fuzzy_mgr}（容错错别字）")
                detected_managers = [fuzzy_mgr]

        detected_strategy = None
        # 先匹配更具体、更长的策略关键词，避免 "A500指增" 被较短的
        # "500指增" 子串提前命中。
        strategy_items = sorted(
            STRATEGY_KEYWORDS.items(),
            # A500 与 500 存在包含关系，A500 必须无条件优先；其余再按关键词长度。
            key=lambda item: (
                item[0] == 'A500指增',
                max(len(keyword) for keyword in item[1]),
            ),
            reverse=True,
        )
        for strategy, keywords in strategy_items:
            if any(kw in question for kw in keywords):
                detected_strategy = strategy
                break

        is_comparison = any(kw in question for kw in COMPARISON_KEYWORDS)
        is_ranking = any(kw in question for kw in RANKING_KEYWORDS)
        return detected_managers, detected_strategy, is_comparison, is_ranking

    def _retrieve_for_question(self, question: str) -> str:
        """按原有优先级路由检索，并返回可直接给模型使用的上下文。

        优先级：管理人对比 → 产品名精确匹配 → 无管理人的策略排名 →
        管理人查询 → 向量检索与重排序。
        """
        (
            detected_managers,
            detected_strategy,
            is_comparison,
            is_ranking,
        ) = self._detect_query_intent(question)

        # Case A：管理人对比。分别扫描后合并，防止只取到其中一家资料。
        if is_comparison and len(detected_managers) >= 2:
            print(f"🔍 [对比查询] 管理人: {detected_managers}")
            all_context = []
            for manager in detected_managers:
                mc = self.retrieve_all_for_manager(manager)
                if mc:
                    all_context.append(mc)
            if all_context:
                return "\n\n---\n\n".join(all_context)

        # Case B：例如“500 指增谁表现好”，没有指定管理人时按策略汇总。
        if detected_strategy and not detected_managers and is_ranking:
            print(f"🔍 [策略排名] 策略: {detected_strategy}")
            context = self.retrieve_by_strategy(detected_strategy)
            if context:
                return context
            context = self.retrieve_with_rerank(question)
            return context or ""

        # 单产品全名比“管理人名”更具体，优先精确命中，避免该管理人的
        # 前 25 条资料把目标产品挤出上下文。策略排名已在上一分支优先处理，
        # 不会把“A500 指增谁好”误当作某个产品名。
        exact_product_context = self.retrieve_by_product_name(question)
        if exact_product_context:
            print("🔍 [精确产品查询] 已优先返回目标产品资料")
            return exact_product_context

        # Case C：单个或多个管理人查询。
        if detected_managers:
            if len(detected_managers) >= 2:
                # 多个管理人同时出现时合并取全部，而非只命中第一个。
                print(f"🔍 [多管理人] {detected_managers}")
                all_context = []
                for manager in detected_managers:
                    mc = self.retrieve_all_for_manager(manager)
                    if mc:
                        all_context.append(mc)
                if all_context:
                    return "\n\n---\n\n".join(all_context)
            print(f"🔍 [单管理人] {detected_managers[0]}")
            context = self.retrieve_all_for_manager(detected_managers[0])
            if context:
                return context
            context = self.retrieve_with_rerank(question, detected_managers[0])
            return context or ""

        # Case D：没有实体且无精确产品名时，回退到通用语义检索。
        print("🔍 [通用查询] 启动向量+Rerank")
        context = self.retrieve_with_rerank(question)
        return context or ""

    def retrieve_context(self, question: str) -> str:
        """兼容流式回答入口：复用统一的检索路由。"""
        return self._retrieve_for_question(question)

    def _build_graph(self):
        app = self

        def retrieve(state: AgentState):
            # 图工作流只负责状态流转；实际检索统一由同一个方法完成。
            return {"context": app._retrieve_for_question(state["question"])}

        def generate(state: AgentState):
            guarded = compliance_guardrail(state['question'])
            if guarded:
                return {"answer": guarded}
            prompt = ChatPromptTemplate.from_template(template)
            chat_history_text = format_chat_history(state.get("chat_history", []))
            chain = prompt | app.llm
            response = chain.invoke({
                "question": state["question"],
                "context": state["context"],
                "chat_history": chat_history_text,
                "data_date": app.data_date or '未知',
            })
            answer = _strip_code_blocks(response.content)

            # 组合配置问题优先走"组合 vs 基准"分析；否则走原有对比/基准图
            portfolio_md = app.generate_portfolio_analysis(state["question"])
            if portfolio_md:
                answer += f"\n\n🧺 **组合收益 vs 基准指数**\n\n{portfolio_md}"
                print("✅ 已生成组合分析")
            else:
                chart_img = generate_chart_from_context(state["context"], state["question"])
                if chart_img:
                    answer += f"\n\n📊 **产品收益对比图**\n\n{chart_img}"
                    print("✅ 已生成对比图表")
                else:
                    bench_md = app.generate_benchmark_chart(state["context"], state["question"])
                    if bench_md:
                        answer += f"\n\n📈 **产品 vs 基准指数**\n\n{bench_md}"
                        print("✅ 已生成基准对比图")

            answer += source_footer(state['context'])

            return {"answer": answer}

        def route_after_retrieve(state: AgentState):
            context = state.get("context", "").strip()
            if context and len(context) > 50:
                print("✅ context 有内容，走 generate")
                return "generate"
            print("❌ context 为空或过短，走 fallback")
            return "fallback"

        def fallback(state: AgentState):
            return {
                "answer": "关于您的问题，我目前的知识库中还没有足够的信息。建议您直接咨询我们的顾问团队。"
            }

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        graph_builder.add_node("fallback", fallback)
        graph_builder.set_entry_point("retrieve")
        graph_builder.add_conditional_edges(
            "retrieve",
            route_after_retrieve,
            {"generate": "generate", "fallback": "fallback"},
        )
        graph_builder.add_edge("generate", END)
        graph_builder.add_edge("fallback", END)
        return graph_builder.compile(checkpointer=InMemorySaver())


_app = RAGApplication()


def get_answer(question: str, session_id: str = "default", knowledge_path='knowledge.txt'):
    _app.initialize(knowledge_path)
    config = {"configurable": {"thread_id": session_id}}

    guarded = compliance_guardrail(question)
    if guarded:
        return guarded

    try:
        current_state = _app.graph.get_state(config)
        previous_chat_history = (
            current_state.values.get("chat_history", [])
            if current_state and current_state.values else []
        )
    except Exception:
        previous_chat_history = []

    updated_chat_history = previous_chat_history + [{"role": "user", "content": question}]
    initial_state = {
        "question": question,
        "chat_history": updated_chat_history,
        "context": "",
        "answer": "",
    }

    final_state = _app.graph.invoke(initial_state, config)
    answer = final_state["answer"]
    final_chat_history = updated_chat_history + [{"role": "assistant", "content": answer}]
    _app.graph.update_state(config, {"chat_history": final_chat_history})
    return answer


def analyze_portfolio(question: str, benchmark_key: Optional[str] = None, knowledge_path='knowledge.txt'):
    """组合分析 UI 入口：返回 (markdown, 可选基准 [(label, key), ...], 默认 key)。
    解析失败时返回 (None, [], None)。"""
    try:
        _app.initialize(knowledge_path)
        data = _app.build_portfolio(question, benchmark_key, require_intent=False)
        if not data:
            return None, [], None
        md = _app.generate_portfolio_analysis(question, benchmark_key, require_intent=False)
        return md, data['bench_options'], data['default_bench_key']
    except Exception as e:
        print(f"⚠️ 组合分析 UI 入口失败：{e}")
        import traceback
        traceback.print_exc()
        return None, [], None


def analyze_client_portfolio(question: str, benchmark_key: Optional[str] = None,
                             knowledge_path='knowledge.txt'):
    """客户版组合分析入口：输入产品列表（未写比例时按等权组合）。
    返回 (markdown, 可选基准 [(label, key), ...], 默认 key)。"""
    try:
        _app.initialize(knowledge_path)
        data = _app.build_portfolio(question, benchmark_key, require_intent=False)
        if not data:
            parsed = parse_portfolio(question)
            names = [name for _, name in parsed] if parsed else (parse_product_list(question) or [])
            missing = [name for name in names if not _app.resolve_portfolio_leg(name)]
            if missing:
                messages = []
                for name in missing:
                    manager = extract_manager(name)
                    if manager in MANAGER_LIST and not _app.list_manager_products(manager):
                        messages.append(
                            f"「{manager}」在知识库中有管理人介绍，但暂未收录任何"
                            "可用于计算的产品业绩数据（策略、区间收益）。"
                        )
                    else:
                        messages.append(f"「{name}」未匹配到可用于计算的产品业绩数据。")
                return "⚠️ **无法生成组合对比**：" + "<br>".join(messages), [], None
            return None, [], None
        md = _app.generate_portfolio_analysis(
            question, benchmark_key, require_intent=False, client_mode=True
        )
        options = [(label.split('（')[0], k) for label, k in data['bench_options']]
        return md, options, data['default_bench_key']
    except Exception as e:
        print(f"⚠️ 客户版组合分析入口失败：{e}")
        import traceback
        traceback.print_exc()
        return None, [], None


def analyze_style(manager_query: str, product_key: Optional[str] = None,
                  index_key: Optional[str] = None, knowledge_path='knowledge.txt'):
    """风格体检 UI 入口：输入管理人/产品名，返回
    (markdown, 产品选项, 默认产品key, 指数选项, 默认指数key)。"""
    try:
        _app.initialize(knowledge_path)
        products = _app.list_manager_products(manager_query)
        if not products:
            manager = manager_query.strip()
            if manager not in MANAGER_LIST:
                manager = _fuzzy_manager_match(manager) or extract_manager(manager)
            if manager in MANAGER_LIST:
                return (
                    f"⚠️ **{manager} 暂无法进行风格体检**：知识库中有管理人资料，"
                    "但没有带区间收益的产品记录；补充产品业绩后即可生成对比图。",
                    [], None, [], None,
                )
        return _app.generate_style_analysis(manager_query, product_key, index_key)
    except Exception as e:
        print(f"⚠️ 风格体检 UI 入口失败：{e}")
        import traceback
        traceback.print_exc()
        return None, [], None, [], None


def stream_answer(question: str, session_id: str = "default", knowledge_path='knowledge.txt'):
    """真流式回答：检索资料后逐 token 输出，并在结束时补充相关图表。"""
    _app.initialize(knowledge_path)
    config = {"configurable": {"thread_id": session_id}}

    guarded = compliance_guardrail(question)
    if guarded:
        yield guarded
        return

    try:
        current_state = _app.graph.get_state(config)
        previous_chat_history = (
            current_state.values.get("chat_history", [])
            if current_state and current_state.values else []
        )
    except Exception:
        previous_chat_history = []

    updated_chat_history = previous_chat_history + [{"role": "user", "content": question}]

    context = _app.retrieve_context(question)
    print(f"📄 检索完成，context 长度: {len(context)} 字符")

    if not context or len(context.strip()) <= 50:
        fallback_msg = "关于您的问题，我目前的知识库中还没有足够的信息。建议您直接咨询我们的顾问团队。"
        final_chat_history = updated_chat_history + [{"role": "assistant", "content": fallback_msg}]
        _app.graph.update_state(config, {"chat_history": final_chat_history})
        yield fallback_msg
        return

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | _app.llm

    # 与 LangGraph 的普通回答保持同一份历史对话截断规则。
    chat_history_text = format_chat_history(updated_chat_history)

    full_answer = ""
    for chunk in chain.stream({
        "question": question,
        "context": context,
        "chat_history": chat_history_text,
        "data_date": _app.data_date or '未知',
    }):
        token = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_answer += token
        yield _strip_code_blocks(full_answer)

    try:
        # 组合配置问题优先走"组合 vs 基准"分析；否则走原有对比/基准图
        portfolio_md = _app.generate_portfolio_analysis(question)
        if portfolio_md:
            full_answer += f"\n\n🧺 **组合收益 vs 基准指数**\n\n{portfolio_md}"
            yield _strip_code_blocks(full_answer)
            print("✅ 组合分析追加成功")
        else:
            chart_img = generate_chart_from_context(context, question)
            if chart_img:
                full_answer += f"\n\n📊 **产品收益对比图**\n\n{chart_img}"
                yield _strip_code_blocks(full_answer)
                print("✅ 图表追加成功")
            else:
                bench_md = _app.generate_benchmark_chart(context, question)
                if bench_md:
                    full_answer += f"\n\n📈 **产品 vs 基准指数**\n\n{bench_md}"
                    yield _strip_code_blocks(full_answer)
                    print("✅ 基准对比图追加成功")
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()

    full_answer += source_footer(context)
    yield _strip_code_blocks(full_answer)

    final_chat_history = updated_chat_history + [{"role": "assistant", "content": full_answer}]
    _app.graph.update_state(config, {"chat_history": final_chat_history})


if __name__ == "__main__":
    _app.initialize()
    print("✅ 系统加载完成")
    if _app.rerank_available:
        print(f"🔍 检索流程：精确匹配 → 向量召回(k={RECALL_K}) → Rerank精排(Top {RERANK_TOP_K})")
    else:
        print(f"🔍 检索流程：精确匹配 → 向量检索(k={RECALL_K}，Rerank不可用，取Top {FALLBACK_TOP_K})")
