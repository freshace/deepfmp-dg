# DeepFMP-DG

**电商多模态预期差诊断。**

DeepFMP-DG 量化商品的“卖家宣传侧”（展示图、描述、价格）与“买家体验侧”（实拍图、评论文本）之间的偏差，预测低评分风险，并通过可解释诊断说明风险来源。

[![CI](https://github.com/freshace/deepfmp-dg/actions/workflows/ci.yml/badge.svg)](https://github.com/freshace/deepfmp-dg/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 项目亮点

- **三维预期差指标**：图-图（卖家展示图 vs 买家实拍图）、文-图（描述 vs 买家实拍图）、文-文（描述 vs 买家评论），统一在 SigLIP 共享语义空间内计算。
- **轻量端到端网络**：双塔视觉编码 + 跨模态注意力 + 差异投影 + Hadamard 交互 + 门控融合头（约 27 万参数，CPU 可推理）。
- **可解释闭环**：风险概率 + TreeSHAP 归因 + 门控权重 + 五级商品诊断与分级处方。
- **可复现**：固定 seed=42、分层 5 折交叉验证、CLI/测试/CI 齐全。

## 实验结果

全量研究数据（3,412 条配对样本，5 折交叉验证，seed=42）：

| 模型 | AUC |
|---|---|
| DeepFMP（视觉） | 0.9069 |
| DeepFMP（视觉+表格） | 0.9456 |
| **DeepFMP（S1+S2+S3 直接拼接）** | **0.9477** |
| DeepFMP（Softmax 门控） | 0.9365 |

复现命令：`python -m deepfmp_dg.cli evaluate --data <建模表.csv> --embeddings <embeddings.npz>`

## 快速开始

```bash
pip install -e ".[dev]"
python -m deepfmp_dg.cli predict \
  --seller-img data/sample/images/seller/<file>.jpg \
  --buyer-img data/sample/images/buyer/<file>.jpg \
  --title "Summer Dress" \
  --description "Lightweight cotton dress" \
  --review "Great quality, fits perfectly!" \
  --price 25.0
```

## 交互演示

```bash
python -m deepfmp_dg.cli demo
```

打开 `http://127.0.0.1:7860`，上传卖家展示图与买家实拍图，即可查看三维预期差、风险等级、SHAP 归因与分级建议。

## 仓库结构

```
src/deepfmp_dg/   # 可安装包（scores / features / models / train / explain / diagnose / infer / cli）
demo/             # Gradio 应用薄包装
examples/         # Notebook 演示
data/sample/      # 小份样例数据 + 演示模型（图片仅供演示）
scripts/          # 样例数据构建与全量复现
tests/            # 单元测试
docs/             # 架构文档 + 模型卡
```

## 与 DeepFMP 的关系

本项目受 **DeepFMP** 启发并做了扩展（Zhang, Ji & Cai, *Clothing Recommendation with Multimodal Feature Fusion: Price Sensitivity and Personalization Optimization*, Applied Sciences 15(8):4591, 2025）。DeepFMP 面向服装搭配推荐，DeepFMP-DG 将多模态融合思想迁移到“单商品预期差诊断”：

| | DeepFMP | DeepFMP-DG |
|---|---|---|
| 任务 | 服装搭配推荐 | 单商品预期差诊断 |
| 输入 | 视觉+文本+价格 | 卖家侧 vs 买家侧配对信息 |
| 机制 | 增强 DeepFM + 注意力 | 三维预期差 + 双塔跨模态注意力 + 门控融合 |
| 输出 | 推荐排序 | 低评分风险 + 可解释诊断 |

本仓库是扩展实现，**非** DeepFMP 官方代码。

## 致谢

- **DeepFMP** — Zhang, Ji & Cai, Applied Sciences 15(8):4591, 2025。
- **SigLIP** — Zhai et al., *Sigmoid Loss for Language Image Pre-Training*。
- **Amazon Reviews 2023** — McAuley Lab 数据集（研究与样例数据来源）。
- 样例图片为公开商品图片，仅供演示（demo-only）。

## 路线图

- FastAPI 推理服务 + Docker
- Hugging Face 权重与数据集托管
- 中文电商（买家秀）迁移实验
- 高风险商品的 VLM 细粒度审核（OCR/VQA）

## 许可

MIT
