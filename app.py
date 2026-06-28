

"""
HCM 患者心血管死亡风险 —— 个体化预测网页计算器（中英双语版）
HCM Cardiovascular Death Risk — Individualized Prediction Web Calculator (Bilingual)
基于随机生存森林 (Random Survival Forest) 模型
"""

import os
import urllib.request
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# 配置 / Configuration
# =============================================================================
MODEL_PATH  = "RandomForest_final.pkl"
MODEL_URL   = "https://huggingface.co/aoxiang199/hcm-rsf-model/resolve/main/RandomForest_final.pkl"
RISK_CUTOFF = 2.324
MONTH_POINTS = [12, 24, 36, 48, 60, 72]

# =============================================================================
# 翻译字典 / Translation Dictionary
# =============================================================================
TRANS = {
    "page_title": {"zh": "HCM心血管死亡风险预测",
                   "en": "HCM Cardiovascular Death Risk Prediction"},
    "title":      {"zh": "肥厚型心肌病（HCM）心血管死亡风险预测",
                   "en": "Hypertrophic Cardiomyopathy (HCM) Cardiovascular Death Risk Prediction"},
    "caption":    {"zh": "基于随机生存森林 (Random Survival Forest) 模型的个体化预测工具",
                   "en": "Individualized prediction tool based on Random Survival Forest model"},

    "switch_to":  {"zh": "Switch to English",
                   "en": "切换为中文"},

    "section_input": {"zh": "请输入患者各项指标",
                      "en": "Enter Patient Parameters"},
    "btn_predict":   {"zh": "开始预测",
                      "en": "Run Prediction"},

    "section_result": {"zh": "预测结果",
                       "en": "Prediction Results"},
    "risk_score":     {"zh": "风险分数 (Risk score)",
                       "en": "Risk Score"},
    "high_risk":      {"zh": f"⚠️ 高风险组\n\n(截断值 = {RISK_CUTOFF})",
                       "en": f"⚠️ High Risk Group\n\n(Cutoff = {RISK_CUTOFF})"},
    "low_risk":       {"zh": f"✅ 低风险组\n\n(截断值 = {RISK_CUTOFF})",
                       "en": f"✅ Low Risk Group\n\n(Cutoff = {RISK_CUTOFF})"},

    "section_curve": {"zh": "个体化生存概率曲线",
                      "en": "Individualized Survival Probability Curve"},
    "xlabel":        {"zh": "随访时间 Follow-up time (months)",
                      "en": "Follow-up Time (months)"},
    "ylabel":        {"zh": "预测生存概率 Predicted Survival Probability",
                      "en": "Predicted Survival Probability"},

    "section_table": {"zh": "关键时间点生存概率",
                      "en": "Survival Probability at Key Time Points"},
    "col_time":      {"zh": "随访时间点",
                      "en": "Follow-up Time"},
    "col_surv":      {"zh": "预测生存概率",
                      "en": "Predicted Survival Probability"},
    "month_unit":    {"zh": "个月",
                      "en": "months"},

    "disclaimer": {"zh": ("⚠️ 本工具基于回顾性队列训练的机器学习模型，"
                          "仅供科研与临床参考，不能替代医生的综合判断，"
                          "结果应结合患者具体临床情况解读。"),
                   "en": ("⚠️ This tool is based on a machine-learning model trained on a "
                          "retrospective cohort. It is intended for research and clinical "
                          "reference only and does not replace comprehensive physician judgment. "
                          "Results should be interpreted in the context of each patient's "
                          "specific clinical situation.")},

    "model_loading": {"zh": "正在首次下载模型文件，约需 30 秒，请稍候...",
                      "en": "Downloading model for the first time, ~30 seconds, please wait..."},
    "model_error":   {"zh": "模型加载失败，请刷新页面重试。\n\n错误详情: {err}",
                      "en": "Failed to load model. Please refresh and try again.\n\nError: {err}"},
}

DISPLAY_LABELS = {
    "age":          {"zh": "年龄 Age (y)",                          "en": "Age (y)"},
    "BMI":          {"zh": "体重指数 BMI (kg/m²)",                  "en": "BMI (kg/m²)"},
    "LVOT":         {"zh": "左室流出道压差 LVOT (mmHg)",            "en": "LVOT Gradient (mmHg)"},
    "LA_dimension": {"zh": "左房内径 LA dimension (mm)",            "en": "LA Dimension (mm)"},
    "LV_diameter":  {"zh": "左室内径 LV diameter (mm)",            "en": "LV Diameter (mm)"},
    "MWT":          {"zh": "最大室壁厚度 MWT (mm)",                 "en": "Max Wall Thickness — MWT (mm)"},
    "LVEF":         {"zh": "左室射血分数 LVEF (%)",                 "en": "LVEF (%)"},
    "CI":           {"zh": "心脏指数 CI (L/min/m²)",               "en": "Cardiac Index — CI (L/min/m²)"},
    "LVEDVi":       {"zh": "左室舒张末容积指数 LVEDVi (mL/m²)",    "en": "LVEDVi (mL/m²)"},
    "LVM_index":    {"zh": "左室质量指数 LVM index (g/m²)",        "en": "LV Mass Index (g/m²)"},
    "LGE_extent":   {"zh": "延迟强化范围 LGE extent (%)",          "en": "LGE Extent (%)"},
    "GLS":          {"zh": "整体纵向应变 GLS (%)",                  "en": "Global Longitudinal Strain — GLS (%)"},
    "GCS":          {"zh": "整体周向应变 GCS (%)",                  "en": "Global Circumferential Strain — GCS (%)"},
}

INPUT_RANGES = {
    "age":          (1.0,   100.0,  50.0,  1.0),
    "BMI":          (10.0,  60.0,   24.0,  0.1),
    "LVOT":         (0.0,   150.0,  20.0,  1.0),
    "LA_dimension": (20.0,  80.0,   40.0,  1.0),
    "LV_diameter":  (20.0,  80.0,   45.0,  1.0),
    "MWT":          (5.0,   50.0,   18.0,  1.0),
    "LVEF":         (5.0,   90.0,   65.0,  1.0),
    "CI":           (0.5,   8.0,    3.0,   0.1),
    "LVEDVi":       (10.0,  250.0,  70.0,  1.0),
    "LVM_index":    (20.0,  400.0,  100.0, 1.0),
    "LGE_extent":   (0.0,   80.0,   5.0,   1.0),
    "GLS":          (-35.0, 5.0,   -15.0,  1.0),
    "GCS":          (-40.0, 5.0,   -20.0,  1.0),
}


# =============================================================================
# 工具函数 / Helpers
# =============================================================================
def t(key: str, lang: str, **kwargs) -> str:
    text = TRANS[key][lang]
    return text.format(**kwargs) if kwargs else text


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        import requests
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return joblib.load(MODEL_PATH)


def predict_rsf_risk(model_obj, X):
    trees = model_obj["trees"]
    if not trees:
        return np.zeros(X.shape[0])
    return np.mean([tr.predict(X) for tr in trees], axis=0)


def predict_rsf_survival_function(model_obj, X, t_grid=None):
    trees = model_obj["trees"]
    if t_grid is None:
        t_grid = model_obj["event_times"]
    mats = []
    for tr in trees:
        sfns = tr.predict_survival_function(X)
        mat  = np.array([fn(t_grid) for fn in sfns])
        mats.append(mat)
    return t_grid, np.mean(mats, axis=0)


# =============================================================================
# 主页面 / Main page
# =============================================================================
def main():
    # ---- 语言状态 / Language state ----
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    lang = st.session_state["lang"]

    st.set_page_config(page_title=t("page_title", lang), layout="centered")

    # ---- 语言切换按钮 / Language toggle ----
    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button(t("switch_to", lang), key="lang_btn"):
            st.session_state["lang"] = "en" if lang == "zh" else "zh"
            st.rerun()
    lang = st.session_state["lang"]

    st.title(t("title", lang))
    st.caption(t("caption", lang))

    # ---- 模型加载 / Load model ----
    try:
        with st.spinner(t("model_loading", lang)):
            bundle = load_model()
    except Exception as e:
        st.error(t("model_error", lang, err=e))
        st.stop()

    model_obj     = bundle["model_obj"]
    feature_names = bundle["feature_names"]

    st.markdown("---")
    st.subheader(t("section_input", lang))

    # ---- 输入表单 / Input form ----
    inputs = {}
    cols = st.columns(2)
    for i, fname in enumerate(feature_names):
        lo, hi, default, step = INPUT_RANGES.get(fname, (0.0, 1000.0, 0.0, 1.0))
        label = DISPLAY_LABELS.get(fname, {}).get(lang, fname)
        with cols[i % 2]:
            inputs[fname] = st.number_input(
                label,
                min_value=float(lo), max_value=float(hi),
                value=float(default), step=float(step),
                format="%.2f",
            )

    st.markdown("---")

    if st.button(t("btn_predict", lang), type="primary", use_container_width=True):
        X = np.array([[inputs[f] for f in feature_names]], dtype=float)

        risk = float(predict_rsf_risk(model_obj, X)[0])

        # 平滑曲线用细密网格 / Dense grid for smooth curve
        plot_grid = np.linspace(0.5, max(MONTH_POINTS), 150)
        t_grid, surv_mat = predict_rsf_survival_function(model_obj, X, t_grid=plot_grid)
        surv = surv_mat[0]

        # 表格用精确时间点 / Exact time points for table
        _, surv_table_mat = predict_rsf_survival_function(
            model_obj, X, t_grid=np.array(MONTH_POINTS, dtype=float))
        surv_table = surv_table_mat[0]

        is_high = risk >= RISK_CUTOFF

        # ---- 风险分组 / Risk group ----
        st.subheader(t("section_result", lang))
        c1, c2 = st.columns(2)
        with c1:
            st.metric(t("risk_score", lang), f"{risk:.3f}")
        with c2:
            if is_high:
                st.error(t("high_risk", lang))
            else:
                st.success(t("low_risk", lang))

        # ---- 生存曲线 / Survival curve ----
        st.subheader(t("section_curve", lang))
        mpl.rcParams["font.family"] = ["DejaVu Sans", "SimHei",
                                        "Arial Unicode MS", "sans-serif"]
        mpl.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(7, 4))
        color = "#C0392B" if is_high else "#2980B9"
        ax.step(t_grid, surv, where="post", color=color, linewidth=2.2)
        ax.fill_between(t_grid, surv, step="post", color=color, alpha=0.08)
        ax.set_xlabel(t("xlabel", lang))
        ax.set_ylabel(t("ylabel", lang))
        ax.set_ylim(0, 1.03)
        ax.set_xlim(0, t_grid.max())
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # ---- 关键时间点表格 / Key time-point table ----
        st.subheader(t("section_table", lang))
        unit = t("month_unit", lang)
        st.table({
            t("col_time", lang): [f"{mo} {unit}" for mo in MONTH_POINTS],
            t("col_surv", lang): [f"{sp * 100:.1f}%" for sp in surv_table],
        })

    st.markdown("---")
    st.caption(t("disclaimer", lang))


if __name__ == "__main__":
    main()
