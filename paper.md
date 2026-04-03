# 分层动态空间图记忆系统

版本 2.0 | 2026年4月

## 摘要

当前大语言模型（LLM）面临上下文窗口限制与静态知识表示的根本困境。
LDSG 通过三层架构（STM/MSG/WS）+ 三锚点动态压缩栈，实现持续学习、渐进遗忘与创造性联想的有机融合。

## 核心创新

1. 空间-语义统一表示：相关性查询 O(1) 复杂度
2. 三层架构：STM（会话）→ MSG（长期）→ WS（工作空间）
3. 三锚点动态压缩栈：多任务并行 + 渐进遗忘
4. 跨锚点桥梁检测：主动发现不同话题间的潜在关联

## 架构

```
STM (会话) → 归并 → MSG (持久图)
                    ↓ 投影 → WS (活跃投影)
                    ↓ 共振 → 三锚点压缩栈
```

## 验证结果

| Phase | 内容 | 状态 |
|-------|------|------|
| P0 | 空间-语义统一 | ✅ 通过 |
| P1 | 关键节点归并 | ✅ 通过 |
| P2 | 双模式漫游 | ✅ 通过 |
| P3-FULL | 共振+晋升+降级 | ✅ 通过 |
| P4 | 端到端整合 | ✅ 通过 |
| P5 | 三锚点+动态压缩栈 | ✅ 通过 |

## 运行

```bash
pip install sentence-transformers numpy
python3 run_p5_full.py
```

## 文件结构

```
src/              # 核心模块
  core_types.py   # Node, SubnodeSignature, ShortTermSubgraph
  encoder.py      # sentence-transformers 封装
  master_graph.py # MasterSpatialGraph
run_p5_full.py   # P5完整验证脚本
docs/             # 论文文档
README.md         # 完整说明
```
