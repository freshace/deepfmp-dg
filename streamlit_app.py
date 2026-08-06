"""Streamlit demo for DeepFMP-DG."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from deepfmp_dg.infer import InferenceEngine

MODEL_DIR = Path(__file__).resolve().parent / "data" / "sample" / "models"


@st.cache_resource(show_spinner="Loading DeepFMP-DG model (first run downloads SigLIP, ~1GB)...")
def get_engine() -> InferenceEngine:
    return InferenceEngine(model_dir=MODEL_DIR)


def save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def main() -> None:
    st.set_page_config(page_title="DeepFMP-DG", page_icon="🔍", layout="wide")
    st.title("DeepFMP-DG — 多模态预期差诊断")
    st.caption("上传卖家展示图、买家实拍图与评论文本，查看三维预期差、低评分风险与诊断建议。")
    st.markdown("> 样例图片仅供演示（demo-only）")

    left, right = st.columns(2)
    seller_file = left.file_uploader("卖家展示图", type=["jpg", "jpeg", "png"])
    buyer_file = right.file_uploader("买家实拍图", type=["jpg", "jpeg", "png"])
    title = st.text_input("商品标题")
    description = st.text_area("商品描述")
    review = st.text_area("买家评论")
    price = st.number_input("价格（可选）", min_value=0.0, value=None, step=1.0)

    if st.button("开始诊断", type="primary"):
        if not seller_file or not buyer_file or not review:
            st.error("请提供卖家图、买家图和评论文本")
            return
        with st.spinner("诊断中..."):
            seller_path = save_upload(seller_file)
            buyer_path = save_upload(buyer_file)
            try:
                result = get_engine().predict(
                    seller_image=seller_path,
                    buyer_image=buyer_path,
                    title=title or "",
                    description=description or "",
                    review_text=review,
                    price=float(price) if price else None,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"预测失败：{exc}")
                return

        scores = result["scores"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Score1 图-图", f"{scores['score1']:.3f}")
        c2.metric("Score2 文-图", f"{scores['score2']:.3f}")
        c3.metric("Score3 文-文", f"{scores['score3']:.3f}")
        st.metric("低评分风险概率", f"{result['risk_probability']:.3f}", result["risk_level"])
        st.subheader("根因诊断")
        for d in result["diagnosis"]:
            st.write(f"- {d}")
        st.subheader("Top-SHAP 特征")
        st.dataframe(
            [
                {"特征": t["display_name"], "SHAP 值": round(t["value"], 4)}
                for t in result["top_shap"]
            ]
        )
        st.subheader("分级建议")
        for r in result["recommendations"]:
            st.write(f"- {r}")


if __name__ == "__main__":
    main()
