# LDSG - 分层动态空间图记忆系统

**Hierarchical Dynamic Spatial Graph Memory System**

一个受认知科学启发的统一记忆框架，实现持续学习、渐进遗忘与创造性联想的有机融合。

[English](#english) | [中文](#中文)

---

## English

### What is LDSG?

LDSG (Hierarchical Dynamic Spatial Graph Memory System) is a cognitively-inspired memory architecture for LLMs that addresses two fundamental limitations:

1. **Context window bottleneck** - Transformer O(n²) self-attention limits context length
2. **Static knowledge** - Knowledge is frozen in model weights, unable to adapt to individual users

### Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Short-Term Memory (STM) - Session-scoped             │
│  Key nodes + Sub-nodes → Merges into MSG after session │
└───────────────────────┬─────────────────────────────────┘
                        │ merge
┌───────────────────────▼─────────────────────────────────┐
│  Master Spatial Graph (MSG) - Persistent               │
│  L3 meta / L2 key / L1 subnode hierarchy               │
│  128-d / 384-d spatial vectors                        │
└───────────────────────┬─────────────────────────────────┘
                        │ project
┌───────────────────────▼─────────────────────────────────┐
│  Working Space (WS) - Active projections              │
│  Resonant detection + Dual-mode wandering             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Three-Anchor Dynamic Compression Stack (NEW in v2)    │
│  Clear Zone (3) / Compressed Zone (6) / Forgetting   │
│  primary 60% / secondary 25% / tertiary 15%          │
└─────────────────────────────────────────────────────────┘
```

### Key Mechanisms

- **Spatial-Semantic Unification**: O(1) similarity search via 128-d unit vectors
- **Hierarchical Node Merging**: Explicit + implicit dual-mode connections
- **Dual-Mode Wandering**: Focused deep-dive (T=0.4) + Idle exploration (T=0.9)
- **Resonance Detection**: Cross-layer subnode activation matching
- **Three-Anchor Compression Stack**: Multi-task parallel with progressive degradation
- **Progressive Forgetting**: Activation decay → clue → forgotten (MSG unchanged)

### Verification Results (P0-P5)

| Phase | Content | Status |
|-------|---------|--------|
| P0 | Spatial-Semantic Unification | ✅ Passed |
| P1 | Key Node Merging | ✅ Passed |
| P2 | Dual-Mode Wandering | ✅ Passed |
| P3-FULL | Resonance + Promotion + Degradation | ✅ Passed |
| P4 | End-to-End Integration | ✅ Passed |
| P5 | Three-Anchor + Compression Stack | ✅ Passed |

### Quick Start

```bash
# Clone
git clone https://github.com/OscarJHLin/ldsg.git
cd ldsg

# Install dependencies
pip install sentence-transformers numpy python-docx

# Run verification
python3 run_p5_full.py

# Generate paper
python3 write_paper.py
```

### Project Structure

```
ldsg/
├── src/                    # Core modules
│   ├── core_types.py       # Node, SubnodeSignature, ShortTermSubgraph
│   ├── encoder.py           # sentence-transformers wrapper
│   └── master_graph.py     # MasterSpatialGraph
├── data/                   # Test data and reports
│   ├── p0-p5/             # Phase verification data
│   └── projection_384_128.npy  # Learned projection matrix
├── run_p5_full.py          # Main verification script
├── write_paper.py           # Paper generation
└── README.md
```

---

## 中文

### 是什么？

LDSG 是一个受认知科学启发的记忆系统，用于增强大语言模型的记忆能力，解决上下文窗口限制和静态知识两大根本困境。

### 核心创新

1. **空间-语义统一表示** — 128维单位向量，O(1) 相似度查询
2. **三层架构** — STM（会话）/ MSG（长期）/ WS（工作空间）
3. **三锚点动态压缩栈** — 多任务并行 + 渐进遗忘 + 思维广度增强
4. **跨锚点桥梁检测** — 主动发现不同话题间的潜在关联

### 验证阶段

- P0: 空间-语义统一 ✅
- P1: 关键节点归并 ✅
- P2: 双模式漫游 ✅
- P3-FULL: 共振检测 + 晋升 + 降级 ✅
- P4: 端到端整合 ✅
- P5: 三锚点 + 动态压缩栈 ✅

### 快速开始

```bash
git clone https://github.com/OscarJHLin/ldsg.git
cd ldsg
pip install sentence-transformers numpy python-docx
python3 run_p5_full.py
```

### 依赖

- Python 3.10+
- sentence-transformers
- numpy
- python-docx (for paper generation)

### 论文

See `论文_LDSG_v2.0_2026-04-03.docx` for the full academic paper (Chinese).

---

## License

MIT License - See LICENSE file.

## 协作者

林筠贺、Kimi

## Citation

```bibtex
@software{ldsg,
  title = {LDSG: Hierarchical Dynamic Spatial Graph Memory System},
  author = {Lin, Junhe and Kimi},
  year = {2026},
  version = {2.0}
}
```
