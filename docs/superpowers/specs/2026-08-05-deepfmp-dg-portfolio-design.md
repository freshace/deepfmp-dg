# DeepFMP-DG 开源 GitHub 项目设计文档

> 日期：2026-08-05
> 状态：已与用户确认（用户批准：双语 README / 样例含少量真实商品图并注明仅供演示 / 命名 deepfmp-dg 并说明对 DeepFMP 的参考 / 仓库内容去求职化，亮点与功能隐式体现能力）

## 1. 背景与目标

基于竞赛项目 DeepFMP-DG（多模态感知视角下的商品“预期差”测度与优化研究）的真实代码、数据与实验结果，重新整理为一个**可直接发布到 GitHub 的开源项目**。

- 项目能力面向：NLP 与多模态算法（内部定位，不写入仓库）。
- 仓库内不出现“比赛”“论文初稿”“案例大赛”“求职”“作品集”“招聘”“面试”等字样；能力通过项目本身的亮点、功能与应用场景**隐式体现**。
- 突出亮点：三维预期差框架、端到端可解释诊断闭环、工程完成度（包结构 + CLI + Demo + 测试 + CI）。

## 2. 项目定位

**DeepFMP-DG: Multimodal Expectation-Gap Diagnosis for E-Commerce**

一个面向电商场景、可解释的多模态预期差诊断项目：

- 用 SigLIP 将“卖家宣传侧”（展示图、描述文案、价格）与“买家体验侧”（实拍图、评论文本）编码到统一向量空间；
- 计算三维预期差 Score1（图-图）、Score2（文-图）、Score3（文-文）；
- 融合视觉 embedding、三维 Score 与 16 维表格特征，训练 DeepFMP-DG 网络预测低评分风险（rating ≤ 2）；
- 输出 SHAP 归因、门控权重与商品级风险画像，形成“预测—归因—诊断—处方”闭环。

目标读者：工程师、研究者与使用者。README 以技术亮点和功能为主，用项目质量本身说话：30 秒内讲清“这是什么、为什么有价值、结果如何、怎么用”。

## 3. 最终交付物

### 3.1 仓库结构

```
deepfmp-dg/
├── README.md                  # 英文主 README
├── README.zh-CN.md            # 中文版
├── LICENSE                    # MIT
├── pyproject.toml             # 包配置 + ruff 规则
├── .gitignore                 # 排除图片/大文件/虚拟环境
├── .github/workflows/ci.yml   # GitHub Actions：pytest + ruff
│
├── src/deepfmp_dg/            # 可安装 Python 包
│   ├── __init__.py
│   ├── scores.py              # SigLIP 编码 + Score1/2/3
│   ├── features.py            # 16 维表格特征
│   ├── models.py              # DeepFMP-DG 网络（双塔/直接拼接/门控）
│   ├── train.py               # seed=42 固定、5 折 CV
│   ├── evaluate.py            # AUC/AP/F1 等指标
│   ├── explain.py             # SHAP + 门控权重
│   ├── diagnose.py            # 商品级风险画像 + 分级处方
│   ├── infer.py               # 单条端到端预测管线
│   └── cli.py                 # 命令行入口
│
├── demo/app.py                # Gradio 交互 Demo
├── examples/demo.ipynb        # 可复现 notebook
├── data/sample/               # 小份样例（~100 条 CSV + 少量示例图 + 预计算 embedding）
├── scripts/                   # 数据获取/全量复现脚本（精选，seed 统一、去死代码）
├── research/                  # 关键原始脚本精选（标注出处）
├── tests/                     # 核心模块单元测试
└── docs/
    ├── architecture.md        # 架构图 + 数据流（Mermaid）
    └── model-card.md          # 模型卡
```

### 3.2 交付物与价值呈现

| 交付物 | 内容 | 呈现的效果 |
|---|---|---|
| 双语 README | 亮点摘要、应用场景、结果表、快速开始、架构图、Roadmap | 30 秒讲清项目价值 |
| 可安装包 | `pip install -e .` 后 `import deepfmp_dg` | 规范的工程结构 |
| CLI | `deepfmp-dg predict ...` → JSON | 开箱即用的命令行体验 |
| Gradio Demo | 上传卖家图/买家图/标题/评论 → 实时诊断 | 直观的交互演示 |
| Notebook | 样例数据全流程复现 | 完整的端到端链路 |
| 测试 + CI | pytest 全绿 + GitHub Actions badge | 可靠的质量保障 |
| 样例数据 + 脚本 | 开箱即用小份数据 + 全量获取说明 | 可复现性 |
| 架构/模型卡文档 | 方法、指标、局限性 | 严谨的技术文档 |

## 4. 已确认的设计决策

1. **命名**：仓库名 `deepfmp-dg`；在 README 中明确对 DeepFMP 的参考关系。
2. **DeepFMP 参考说明**：README 增加 “Relation to DeepFMP / Acknowledgements” 段落：
   - DeepFMP 论文引用：Zhang, Ji & Cai, *Clothing Recommendation with Multimodal Feature Fusion: Price Sensitivity and Personalization Optimization*, Applied Sciences 15(8):4591, 2025；
   - SigLIP（Zhai et al.）与 Amazon Reviews 2023（McAuley Lab）数据来源；
   - 声明本项目是“面向预期差诊断任务的扩展实现，非官方代码”；
   - 用对比表说明任务/输入/机制/输出的差异（推荐 vs 诊断）。
3. **语言**：README 中英双语（英文主文档 + 中文版）。
4. **样例数据**：放少量真实商品图并注明“仅供演示”（demo-only），不放全量图片。
5. **档位**：第 2 档（包 + CLI + Demo + notebook）+ 第 3 档的轻量信号（pytest + CI）。
6. **明确不做**：FastAPI 服务、Docker、Hugging Face 权重托管——写入 Roadmap。
7. **内容中性化**：README 与全部文档只讲项目亮点、功能与应用场景，不出现任何求职/作品集/招聘相关表述；能力由项目本身隐式体现。

## 5. 架构与数据流

```
输入（卖家图/买家图/标题/描述/评论/价格）
  → scores.py: SigLIP 编码 + 三维余弦 Score
  → features.py: 16 维表格特征
  → models.py: DeepFMP-DG 前向（视觉双塔 + 跨模态注意力 + 对齐/表格分支 + 门控融合）
  → train.py/evaluate.py: seed=42 5 折 CV，AUC/AP/F1
  → explain.py: SHAP 全局/局部归因 + 门控权重
  → diagnose.py: 商品级风险画像 + 五级风险 + 分级处方
  → cli.py / demo/app.py: JSON 输出 / 交互界面
```

模块职责单一、接口明确：

- `scores.py`：图片/文本加载、SigLIP 编码（缓存 embedding）、余弦相似度；
- `features.py`：从原始字段构造建模所需 16 维特征（含文本情感、价格分位、交互项）；
- `models.py`：纯 PyTorch 网络定义（双塔编码器、双向跨模态注意力、差异投影、Hadamard 交互、Softmax 门控、分类头）；
- `train.py`：数据加载、固定 seed、5 折 CV、训练循环、模型保存；
- `evaluate.py`：AUC/AP/F1/Precision/Recall/Accuracy；
- `explain.py`：TreeSHAP + 门控权重提取；
- `diagnose.py`：商品级聚合、五级风险分级、处方模板；
- `infer.py`：单条样本端到端管线，返回结构化结果；
- `cli.py`：`deepfmp-dg` 命令入口（`predict` / `demo` / `train` / `eval`）。

### CLI JSON 输出示例

```json
{
  "scores": {"score1": 0.54, "score2": -0.03, "score3": 0.62},
  "risk_probability": 0.87,
  "risk_level": "高",
  "top_shap": [["评论情感差异", 0.1244], ["预期差×否定词", 0.1074]],
  "diagnosis": ["视觉欺骗：展示图过度美化", "质量缺陷：品质承诺未兑现"],
  "recommendations": ["P0: 平台立即启动信息审核", "P1: 更换无修图主图"]
}
```

## 6. 错误处理与边界

- 图片缺失/加载失败：返回明确错误信息，不崩溃；
- 无 GPU：自动回落 CPU；
- SigLIP 首次运行：自动从 Hugging Face 下载（README 说明）；
- Demo：默认加载样例数据，离线可玩；
- 字段缺失：结构化输出中给出提示而非异常；
- 统一 JSON 接口，便于 CLI/Demo/测试复用。

## 7. 测试与 CI

- `tests/test_scores.py`：余弦值域、归一化、已知样例一致性；
- `tests/test_features.py`：与已知样例值一致；
- `tests/test_models.py`：前向 shape、seed 复现性；
- `tests/test_cli.py`：端到端冒烟测试（predict 返回合法 JSON）；
- CI：`.github/workflows/ci.yml` 运行 `pytest` + `ruff check`。

## 8. 数据与许可

- 代码：MIT License；
- 样例数据：~100 条 CSV（脱敏字段子集）+ 少量真实商品图（注明 demo-only + 来源 Amazon 商品图，仅供演示）；
- 全量数据：`scripts/` 提供 Amazon Reviews 2023 获取与预处理说明，不提交大文件；
- 第三方归因：DeepFMP 论文、SigLIP、Amazon Reviews 2023 在 README/docs 中完整标注；
- 仓库不含：全量图片、全量 embedding、训练中间产物、.venv。

## 9. 范围外（Roadmap 内容）

- FastAPI 推理服务；
- Docker 化部署；
- Hugging Face 权重/数据集托管；
- 中文电商场景（淘宝/拼多多买家秀）迁移验证；
- 与图文一致性检测基线（如 CLIP/BLIP 方案）的对比实验。

## 10. 成功标准

1. clone → `pip install -e .` → `deepfmp-dg demo` 离线可玩；
2. `deepfmp-dg predict ...` 返回合法 JSON；
3. `pytest` 全绿，CI badge 可用；
4. README 30 秒讲清亮点、场景与结果；
5. 仓库无比赛字样、无求职/作品集相关表述、无大文件、无死代码；
6. 模型权重策略：若现有 checkpoint 体积小（< 50MB）且可用则收录，否则提供一键训练脚本 + 复现结果表（0.9477 AUC）。

## 11. 风险与应对

| 风险 | 应对 |
|---|---|
| 图片版权 | 仅保留 2-4 张示例图 + demo-only 声明 |
| 仓库体积 | 样例数据小份化；大文件全部 .gitignore |
| 原代码质量 | 重写核心为模块化包，原始脚本精选进 research/ |
| 复现性 | 全部固定 seed=42，README 给出复现命令 |
| 模型权重缺失 | 训练脚本 + 结果表兜底 |

