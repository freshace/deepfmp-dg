"""Interactive Gradio demo for single-item expectation-gap diagnosis."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from deepfmp_dg.infer import DEFAULT_MODEL_DIR, InferenceEngine

PathLike = str | Path


def _format_result(result: dict) -> str:
    scores = result["scores"]
    lines = [
        "## 诊断结果",
        "",
        f"**Score1（图-图视觉预期差）**: {scores['score1']:.3f}",
        f"**Score2（文-图预期差）**: {scores['score2']:.3f}",
        f"**Score3（文-文预期差）**: {scores['score3']:.3f}",
        "",
        f"**低评分风险概率**: {result['risk_probability']:.3f}",
        f"**风险等级**: {result['risk_level']}",
        "",
        "**根因诊断**:",
    ]
    lines += [f"- {c}" for c in result["diagnosis"]]
    lines += ["", "**Top-SHAP 特征**:"]
    lines += [f"- {t['display_name']}（{t['value']:+.4f}）" for t in result["top_shap"]]
    lines += ["", "**分级建议**:"]
    lines += [f"- {r}" for r in result["recommendations"]]
    return "\n".join(lines)


def _make_handler(engine: InferenceEngine):
    def diagnose(seller_img, buyer_img, title, description, review, price):
        if not seller_img or not buyer_img or not review:
            return "请提供卖家图、买家图和评论文本"
        try:
            result = engine.predict(
                seller_image=seller_img,
                buyer_image=buyer_img,
                title=title or "",
                description=description or "",
                review_text=review,
                price=float(price) if price else None,
            )
            return _format_result(result)
        except Exception as exc:  # noqa: BLE001
            return f"预测失败：{exc}"

    return diagnose


def launch(model_dir: PathLike = DEFAULT_MODEL_DIR, share: bool = False) -> None:
    engine = InferenceEngine(model_dir)
    handler = _make_handler(engine)

    with gr.Blocks(title="DeepFMP-DG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# DeepFMP-DG\n\n"
            "多模态预期差诊断：上传卖家展示图、买家实拍图、标题/描述与评论文本，"
            "实时输出三维预期差、低评分风险、SHAP 归因与分级建议。\n\n"
            "> 样例图片仅供演示（demo-only），来自公开商品页面。"
        )
        with gr.Row():
            seller_input = gr.Image(type="filepath", label="卖家展示图")
            buyer_input = gr.Image(type="filepath", label="买家实拍图")
        title_input = gr.Textbox(label="商品标题")
        desc_input = gr.Textbox(label="商品描述")
        review_input = gr.Textbox(label="买家评论", lines=4)
        price_input = gr.Number(label="价格（可选）", value=None)
        button = gr.Button("开始诊断", variant="primary")
        output = gr.Markdown()
        button.click(
            handler,
            inputs=[seller_input, buyer_input, title_input, desc_input, review_input, price_input],
            outputs=output,
        )
    demo.launch(share=share)


if __name__ == "__main__":
    launch()
