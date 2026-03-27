# titans_lin 训练分支协作说明

本文件用于约束当前训练分支在 **Linux 训练服务器** 上的工作方式。

目标是：

- 本地开发代码
- 推送到服务器 `~/lin/` 目录训练
- 使用 `conda` 环境 `titans_train`
- 训练框架统一为 `accelerate + DDP/FSDP`
- 当前主用 backbone 为 `Qwen3.5-7B`
- 后续需支持快速切换到 `MiniMax` 路线

---

## 1. 当前分支定位

本目录是 Titans 训练实验分支，用于实现：

- 冻结大模型 backbone
- Titans memory 作为可训练外部记忆
- 基于 MAC（Memory as Context）思路构建长程记忆训练方案

当前默认 backbone：

- `Qwen3.5-7B`

后续预留切换能力：

- `MiniMax`

因此实现时必须避免把训练逻辑写死在单一 backbone 上。

---

## 2. 工作目录与同步方式

### 本地开发目录

- 当前工作目录：`/mnt/e/tool/gitlearn/titans_lin`

### 训练服务器连接方式

- SSH 地址：`ssh shuzuan@111.6.70.85 -p 101`

使用原则：

1. 服务器主要用于正式训练与结果保存
2. 不把服务器当长期手工开发环境
3. 服务器上的临时改动应及时同步回本地

### 服务器训练目录

建议统一使用：

- `~/lin/titans_lin/`

### 工作原则

1. 本地负责代码修改、脚本整理、配置管理
2. 服务器负责正式训练、日志记录、checkpoint 保存
3. 不在服务器上随意进行大范围临时改代码
4. 如需服务器侧临时修复，必须尽快同步回本地分支

---

## 3. 环境约束

### conda 环境

训练统一使用：

- `titans_train`

示例：

```bash
conda create -n titans_train python=3.10 -y
conda activate titans_train
```

### 环境原则

1. 不混用多个训练环境
2. 不依赖系统 python 直接训练
3. 所有正式训练命令都在 `titans_train` 内执行

---

## 4. 训练框架约束

统一采用：

- `accelerate launch`

分布式模式：

- `DDP`
- `FSDP`

原则：

1. 不单独维护 `torchrun` 专用训练逻辑
2. 训练脚本应天然兼容 DDP/FSDP
3. 配置切换通过 `accelerate` config 完成，而不是改训练主脚本

---

## 5. 模型训练约束

### 当前方案

- backbone：`Qwen3.5-7B`
- backbone 参数：**全部冻结**
- 不使用 LoRA
- 仅训练 Titans memory 相关参数

### MAC 定义

此处的 MAC 指 Titans 架构中的 **Memory as Context**，不是设备或平台含义。

训练目标是：

1. 从历史 chunk 中写入长期记忆
2. 在问题阶段从 memory 检索相关信息
3. 将 memory 检索结果作为上下文前缀喂给冻结 backbone

---

## 6. 代码设计约束

### 必须遵守

1. 不直接把当前 `src/titans/models.py` 中的 `TitansMAC` 当作最终训练 backbone
2. 应采用“冻结 backbone + Titans memory hybrid”结构
3. backbone 适配层应尽量模块化，避免后续切换 `MiniMax` 时大面积重写

### 推荐方向

- `src/titans/backbones/` 目录下放不同 backbone 适配器
- memory 逻辑保持独立
- 训练主脚本依赖统一接口，而不是依赖某一个特定模型类

---

## 7. 数据集约束

第一阶段数据集采用 **chunk 化合成记忆数据**，主要验证：

1. memory 是否学会写入重要事实
2. memory 是否能在后续 query 中检索出正确信息
3. 冻结 backbone 是否能利用 memory context 回答问题

第一阶段优先任务：

- 单事实召回
- 多实体同属性干扰
- 更新覆盖

第一阶段暂不优先：

- 复杂多跳推理
- 开放式长答案生成
- 大规模真实长文预训练

---

## 8. 服务器执行规范

### 训练前

1. 确认当前目录为 `~/lin/titans_lin/`
2. 激活 `conda activate titans_train`
3. 检查 `accelerate` 配置
4. 检查输出目录、日志目录、checkpoint 目录

### 训练时

1. 优先通过 `accelerate launch --config_file ...` 启动
2. 保留完整日志
3. 重要实验必须记录使用的数据配置、模型配置、checkpoint 路径

### 训练后

1. 记录最终 checkpoint 位置
2. 记录关键指标
3. 如服务器上做了临时代码修改，及时同步回本地

---

## 9. 输出目录建议

建议统一目录结构：

```text
~/lin/titans_lin/
├── configs/
├── scripts/
├── src/
├── data/
├── checkpoints/
├── logs/
└── outputs/
```

建议：

- 训练日志放 `logs/`
- checkpoint 放 `checkpoints/`
- 导出的分析结果放 `outputs/`

---

## 10. checkpoint 约束

建议同时支持两类 checkpoint：

### memory-only checkpoint

保存：

- Titans memory 参数
- persistent memory 参数
- backbone 适配必要配置

### full training checkpoint

保存：

- optimizer state
- scheduler state
- accelerator/FSDP 恢复状态
- 全部训练元信息

原则：

1. 日常实验优先 memory-only
2. 长训练任务需要 full checkpoint 以支持恢复

---

## 11. backbone 切换要求

当前主用：

- `Qwen3.5-7B`

后续目标：

- `MiniMax`

因此实现时必须满足：

1. backbone 加载逻辑可替换
2. tokenizer / processor 接口可替换
3. hidden size / embedding 维度映射可配置
4. 训练脚本尽量不感知具体 backbone 细节

不允许把以下信息硬编码在多个位置：

- 模型名称
- hidden size
- tokenizer 路径
- 特定模型专属字段名

---

## 12. 与 Claude 协作规则

### 允许 Claude 做的事

1. 修改当前分支代码
2. 新增训练脚本、配置文件、数据脚本
3. 优化分布式训练入口
4. 完善 SPEC / 操作文档

### 需要谨慎的事

1. 不要随意改动与当前训练路线无关的大块代码
2. 不要默认删除已有训练脚本，除非确认不再使用
3. 不要在未确认前引入额外复杂训练策略（如 LoRA、全参微调）

### 文档规则

除非用户明确要求，否则不要自动记录讨论内容到其他文档。

---

## 13. 当前优先级

当前优先级从高到低如下：

1. 明确 Frozen Backbone + Titans MAC hybrid 的代码结构
2. 完成第一阶段合成数据生成脚本
3. 完成 `accelerate + DDP/FSDP` 训练入口
4. 跑通 Qwen3.5-7B 冻结训练闭环
5. 为后续切换 MiniMax 预留适配层

---

## 14. 当前结论

本分支当前的核心路线是：

- 在 Linux 服务器上训练
- 使用 `titans_train` conda 环境
- 使用 `accelerate + DDP/FSDP`
- 以冻结 backbone 为前提
- 只训练 Titans memory
- 当前主用 Qwen3.5-7B
- 后续支持快速切换到 MiniMax

这是当前目录下后续开发、训练、文档整理的共同约束。

