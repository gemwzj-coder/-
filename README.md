# 投研知识库智能工作台

一个面向投研运营场景的 RAG（检索增强生成）Demo。它将产品、管理人、策略与业绩资料组织为可检索知识库，并提供流式问答、客观对比、组合/风格分析、知识库版本管理和回归评测能力。

> 本项目用于展示 RAG 产品设计与工程实现。公开仓库只包含脱敏样例数据；不提供个性化投资建议，也不应作为投资决策依据。

## 作品亮点

- **面向业务的检索路由**：按产品、管理人、策略和对比类问题选择检索路径，再结合向量检索与重排序。
- **可解释的结果呈现**：流式输出答案，并按问题类型补充业绩、基准或组合图表。
- **知识库治理**：内容哈希、版本快照、索引重建、回滚和审计记录。
- **可回归评测**：支持检索命中、事实一致性、引用完整性与合规行为等维度。
- **演示友好界面**：Gradio 多标签页覆盖问答、组合分析、风格分析与管理入口。

## 架构概览

```text
知识库文本 → 解析与切分 → Embedding → Chroma 向量库
                                      ↓
用户问题 → 意图识别/规则路由 → 召回 + Rerank → DeepSeek → 流式回答 / 图表
                                      ↓
                              评测、版本、审计与回滚
```

## 快速开始

建议使用 Python 3.11 或更高版本。

```powershell
git clone <your-repository-url>
cd ai

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 使用可公开的脱敏样例数据
Copy-Item knowledge.example.txt knowledge.txt

# 填写自己的 DeepSeek Key；.env 已被 Git 忽略
Copy-Item .env.example .env
notepad .env

python app.py
```

启动后访问终端输出的本地地址（通常为 `http://127.0.0.1:7860`）。首次运行会下载 Embedding/Rerank 模型并构建向量库，耗时会比后续启动长。

`DEEPSEEK_API_KEY` 既可以写在被忽略的 `.env` 中，也可以通过系统环境变量设置；系统环境变量优先。旧的 `local_config.py` 仍可兼容，但不建议在新环境继续使用。

## 建议

1. 在“产品检索与对比”页询问示例产品的基本情况和业绩，展示检索结果与引用。
2. 在“组合 / 风格分析”页展示结构化输入如何产出对比图表。
3. 打开“管理台”，说明知识库哈希、重建、版本快照、回滚和回归评测如何降低内容更新风险。
4. 强调系统的边界：它提供资料检索与客观比较，而非个性化投资建议。

公开仓库内的 `knowledge.example.txt` 使用虚构数据，只用于演示。真实知识库、评测集、Chroma 索引、模型缓存和历史报告均被 `.gitignore` 排除。提交前请确认你对任何演示数据拥有公开和再分发的权限。

## 常用命令

```powershell
# 查看知识库版本和索引状态
python manage_kb.py status

# 校验知识库结构、重复问题和数值冲突
python manage_kb.py validate

# 重建向量库（知识库更新后执行）
python manage_kb.py rebuild --note "update demo data"

# 运行回归评测；需要本地私有评测集
python evaluate_rag.py --suite all

# 运行基础测试
python -m unittest discover -s tests -v
```

## 项目结构

```text
├── app.py                     # Gradio 演示界面
├── rag_graph业绩.py            # RAG 流程、检索路由、图表与分析能力
├── kb_management.py           # 知识库校验、版本、审计与状态管理
├── manage_kb.py               # 知识库管理命令行入口
├── evaluate_rag.py            # 回归评测脚本
├── knowledge.example.txt      # 可公开的脱敏样例数据
├── tests/                     # 标准库单元测试
├── requirements.txt           # 运行依赖
└── .github/workflows/         # GitHub Actions 基础检查
```

## 发布到 GitHub 前

安装 Git for Windows 后，在项目根目录执行：

```powershell
git init
git add .
git status
git commit -m "feat: publish RAG research workbench demo"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

提交前务必检查 `git status`：不应看到 `.env`、`local_config.py`、`knowledge.txt`、`chroma_db*`、`cache/`、简历产物或日志。若密钥曾经进入任意 Git 历史，请先在服务商后台轮换该密钥。

## License

源代码以 [MIT License](LICENSE) 发布。该许可不自动覆盖任何真实业务数据、第三方数据或商标；它们应按各自的授权条款处理。
