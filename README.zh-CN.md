# CS336 实验一：语言模型基础

<p align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E4%B8%AD%E6%96%87-red?style=for-the-badge" alt="中文" />
    </a>
    &nbsp;&nbsp;
    <a href="./README.md">
        <img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge" alt="English" />
    </a>
</p>

本仓库包含我已完成的 CS336 Lab 1 全部实现。
项目从零构建了一个轻量级 GPT 风格语言模型流程，包括分词器训练、Transformer 核心模块、优化器与训练工具、训练与评估、检查点保存以及文本生成。

## 项目亮点

- 实现了 BPE 分词器训练与持久化
- 实现了数据分词与二进制内存映射数据集构建
- 从零实现 Transformer 关键模块：
  - 词嵌入
  - 多头自注意力
  - RoPE 位置编码
  - RMSNorm
  - SwiGLU 前馈网络
- 实现了自定义 AdamW 优化器
- 实现了学习率调度与梯度裁剪
- 实现了包含验证、检查点、可选 WandB 记录的训练流程
- 实现了自回归文本生成脚本

## 项目结构

- `cs336_basics/tokenizer/`: 分词器实现与 BPE 训练
- `cs336_basics/module/`: 神经网络模块（注意力、归一化、Transformer Block、语言模型）
- `cs336_basics/optimizer/`: 自定义优化器
- `cs336_basics/utils/`: 训练与数学工具函数
- `cs336_basics/load/`: 数据加载与批处理
- `cs336_basics/checkpoint/`: 检查点保存与加载
- `cs336_basics/main.py`: 训练入口
- `cs336_basics/prepare_tokens.py`: 训练分词器并生成 token 文件
- `cs336_basics/generate.py`: 基于检查点生成文本
- `tests/`: 课程测试与样例数据

## 环境配置

本项目使用 `uv` 管理依赖与环境。

安装 `uv`：

```bash
# Linux/macOS（推荐）
# 参考: https://docs.astral.sh/uv/getting-started/installation/

# 可选方式
pip install uv
```

安装依赖：

```bash
uv sync
```

在项目环境中执行命令：

```bash
uv run <command>
```

## 数据准备

将 TinyStories 训练集和验证集下载到 `data/` 目录：

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

## 快速开始

### 1) 运行单元测试

```bash
uv run pytest
```

### 2) 训练分词器并生成 token 二进制文件

```bash
uv run python -m cs336_basics.prepare_tokens
```

默认输出文件：

```text
data/tinystories_vocab.json
data/tinystories_merges.txt
data/train_token_origin.bin
data/valid_token_origin.bin
```

如果希望直接使用训练脚本默认参数，请确保 token 文件名匹配：

```text
data/train_token.bin
data/valid_token.bin
```

### 3) 训练语言模型

```bash
uv run python -m cs336_basics.main \
  --device cpu \
  --iters 20000 \
  --batch-size 64
```

常用参数：

```text
--checkpoint-dir
--iters-per-checkpoint
--iters-per-evaluation
--wandb-project
--train-tokens-filepath
--valid-tokens-filepath
```

### 4) 使用检查点生成文本

```bash
uv run python -m cs336_basics.generate "Once upon a time" \
  --checkpoint-path ./checkpoint/best_checkpoint_iter_20000.pt \
  --max-new-tokens 200 \
  --temperature 0.8
```

## 复现说明

- Python 版本要求：3.11+
- 依赖定义在 `pyproject.toml`
- 检查点默认保存在 `checkpoint/`
- 训练流程包含周期性验证与最优检查点保存

## 实现覆盖范围

本次提交覆盖了 Lab 1 的全部核心模块：

- 分词器训练与编码流程
- Transformer 模块与端到端语言模型
- 损失函数、优化器、学习率调度与梯度裁剪
- 完整训练与评估流程
- 检查点存取与文本生成

## 致谢

- 感谢 Stanford CS336 课程团队提供作业设计与测试框架
- 感谢 TinyStories 数据集作者与托管平台
