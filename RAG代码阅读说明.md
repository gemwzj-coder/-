# RAG 业绩模块阅读说明

主文件是 `rag_graph业绩.py`。它并不是只有“问答”，而是把知识库解析、资料检索、业绩图表和组合分析都放在一起，所以文件较长。建议按下面顺序阅读。

## 一句话流程

用户在 `app.py` 输入问题 → `stream_answer()` 保存对话历史 → `_retrieve_for_question()` 找资料 → DeepSeek 根据资料生成文字 → 视问题补充业绩图 / 基准图 / 组合图 → Gradio 逐字显示答案。

## 文件中最值得看的部分

| 位置 | 作用 | 阅读重点 |
|---|---|---|
| 文件开头的常量 | 管理人名单、策略关键词、图表区间 | 新增管理人或策略时从这里改 |
| `load_knowledge_qa()` | 将 `knowledge.txt` 切分成文档 | 知识库格式不对时看这里 |
| `RAGApplication.initialize()` | 加载嵌入模型、Chroma 向量库、重排序模型和 DeepSeek | 首次启动或知识库重建时执行 |
| `_detect_query_intent()` | 识别管理人、策略、对比/排名意图 | 决定问题属于哪一类 |
| `_retrieve_for_question()` | 统一检索路由 | 这是问答最关键的方法 |
| `generate_*chart*()` | 从资料中提取业绩并画图 | “为什么没有今年以来”优先检查这一组 |
| `stream_answer()` | 流式回答的公开入口 | `app.py` 实际调用这里 |

## 检索路由（已统一）

```text
问题
 ├─ 提到两家及以上管理人且有“对比”等词 → 分别扫描每家资料，再合并
 ├─ 提到策略且有“谁更好/排名”等词、未提管理人 → 按策略汇总
 ├─ 提到管理人 → 优先全量扫描该管理人的资料，未命中再向量检索
 └─ 未提明确实体 → 产品名精确匹配，未命中再向量检索 + Rerank
```

过去普通回答和流式回答各自实现了一份上述路由；现在它们都调用 `_retrieve_for_question()`，以后改规则只需要改一处。

## 与网页的关系

- `app.py`：页面、聊天展示、管理员知识库面板。
- `rag_graph业绩.py`：RAG 核心、图表、组合/风格分析。
- `kb_management.py`：知识库指纹、版本、重建和校验。
- `knowledge.txt`：实际喂给 RAG 的问答资料，不是 Python 代码。

## 最常见的排查位置

| 现象 | 先看哪里 |
|---|---|
| 回答“Connection error” | `DEEPSEEK_API_KEY`、网络/代理、DeepSeek 服务可达性 |
| 问题答非所问 | `_detect_query_intent()` 和 `_retrieve_for_question()` 的命中日志 |
| 图里没有“今年以来” | `knowledge.txt` 的原始业绩字段、`SERIES_BUCKETS` 与图表解析函数 |
| 更新资料后答案还是旧的 | 管理员面板重建知识库，或检查 `kb_management.py` 的状态 |

## 密钥配置

密钥不再写在源码中。Windows PowerShell 可执行以下命令后**关闭并重新打开终端**：

```powershell
[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', '？', 'User')
```

不要把密钥提交到 Git、截图或发到聊天窗口。
