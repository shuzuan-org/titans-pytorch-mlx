# 记忆体系重构：L1/L2/L3 职责边界

## 重构前的问题

- L1（memory/）混入了大量具体数值、命令、参考信息
- L2（CLAUDE.md）只有 SSH 接入，缺命令和配置
- L3 不存在

## 重构后的结构

```
L1  私有，自动注入（memory/）
├── MEMORY.md       当前阶段、架构直觉、未解问题
├── decisions.md    决策日志（做了什么、为什么、何时重新评估）
└── bottlenecks.md  已知瓶颈根因与状态

L2  git-tracked（CLAUDE.md）
    项目目标、操作命令、模型规格、TPTT 配置、文件索引

L3  git-tracked（docs/）
├── log/            日期目录，每天多个事件文件（本文件所在位置）
├── titans-paper.md 论文结论与后续工作
├── model-specs.md  基模架构规格
└── implementation-notes.md 代码实现细节
```

## 归属规则（MECE）

| 信息类型 | 层 |
|---------|---|
| 能从代码/文档查到的具体值 | L2 或 L3 |
| 操作命令、步骤 | L2 |
| "为什么这样设计"的判断 | L1 |
| 今天发生了什么、实验结果 | L3/log/日期/ |

## log 系统工作方式

- 有值得记录的事件 → 在 `docs/log/YYYY-MM-DD/` 下新建一个 `.md` 文件
- 文件名描述事件内容（kebab-case）
- L1/decisions.md 或 MEMORY.md 引用时，直接写路径：`docs/log/2026-03-03/tptt-decision.md`
- 没有 index 文件，L1 就是索引层
