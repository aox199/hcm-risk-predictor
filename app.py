# # -*- coding: utf-8 -*-
# """
# HCM 患者心血管死亡风险 —— 个体化预测网页计算器
# 基于随机生存森林 (Random Survival Forest) 模型
#
# 运行方式（本地测试）：
#     streamlit run app.py
# """
#
# import joblib
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
#
# # =============================================================================
# # 配置
# # =============================================================================
# MODEL_PATH = "RandomForest_final.pkl"   # 部署时和 app.py 放在同一目录
# RISK_CUTOFF = 2.324                      # X-tile 分析得到的高/低风险截断值
# MONTH_POINTS = [12, 24, 36, 48, 60, 72]  # 与训练时的评估时间点一致
#
# # 变量显示名称（中英对照 + 单位），key 必须和 pkl 里 feature_names 的命名完全一致
# DISPLAY_LABELS = {
#     "age":          "年龄 Age (y)",
#     "BMI":          "体重指数 BMI (kg/m²)",
#     "LVOT":         "左室流出道压差 LVOT (mmHg)",
#     "LA_dimension": "左房内径 LA dimension (mm)",
#     "LV_diameter":  "左室内径 LV diameter (mm)",
#     "MWT":          "最大室壁厚度 MWT (mm)",
#     "LVEF":         "左室射血分数 LVEF (%)",
#     "CI":           "心脏指数 CI (L/min/m²)",
#     "LVEDVi":       "左室舒张末容积指数 LVEDVi (mL/m²)",
#     "LVM_index":    "左室质量指数 LVM index (g/m²)",
#     "LGE_extent":   "延迟强化范围 LGE extent (%)",
#     "GLS":          "整体纵向应变 GLS (%)",
#     "GCS":          "整体周向应变 GCS (%)",
# }
#
# # 输入范围: (最小值, 最大值, 默认值, 步长) —— 范围设得比较宽松，避免限制住真实患者数值
# INPUT_RANGES = {
#     "age":          (1.0,   100.0, 50.0,  1.0),
#     "BMI":          (10.0,  60.0,  24.0,  0.1),
#     "LVOT":         (0.0,   150.0, 20.0,  1.0),
#     "LA_dimension": (20.0,  80.0,  40.0,  1.0),
#     "LV_diameter":  (20.0,  80.0,  45.0,  1.0),
#     "MWT":          (5.0,   50.0,  18.0,  1.0),
#     "LVEF":         (5.0,   90.0,  65.0,  1.0),
#     "CI":           (0.5,   8.0,   3.0,   0.1),
#     "LVEDVi":       (10.0,  250.0, 70.0,  1.0),
#     "LVM_index":    (20.0,  400.0, 100.0, 1.0),
#     "LGE_extent":   (0.0,   80.0,  5.0,   1.0),
#     "GLS":          (-35.0, 5.0,   -15.0, 1.0),
#     "GCS":          (-40.0, 5.0,   -20.0, 1.0),
# }
#
#
# # =============================================================================
# # 模型加载与预测函数（逻辑与训练脚本中 predict_risk / predict_survival_function
# # 针对 type=="RandomForest" 的部分完全一致）
# # =============================================================================
# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)
#
#
# def predict_rsf_risk(model_obj, X):
#     trees = model_obj["trees"]
#     if not trees:
#         return np.zeros(X.shape[0])
#     return np.mean([tr.predict(X) for tr in trees], axis=0)
#
#
# def predict_rsf_survival_function(model_obj, X, t_grid=None):
#     """
#     t_grid 可以自定义传入：每棵树内部的生存函数本身是完整保留的，
#     并不受 model_obj["event_times"] 里采样点稀疏与否的影响，
#     所以这里允许传入任意细密的时间网格，用于画出平滑曲线。
#     不传则默认用 model_obj["event_times"]（用于和训练脚本逻辑保持一致）。
#     """
#     trees = model_obj["trees"]
#     if t_grid is None:
#         t_grid = model_obj["event_times"]
#     mats = []
#     for tr in trees:
#         sfns = tr.predict_survival_function(X)
#         mat = np.array([fn(t_grid) for fn in sfns])
#         mats.append(mat)
#     avg = np.mean(mats, axis=0)
#     return t_grid, avg
#
#
# # =============================================================================
# # 页面
# # =============================================================================
# def main():
#     st.set_page_config(page_title="HCM心血管死亡风险预测", layout="centered")
#
#     st.title("肥厚型心肌病（HCM）心血管死亡风险预测")
#     st.caption("基于随机生存森林 (Random Survival Forest) 模型的个体化预测工具")
#
#     try:
#         bundle = load_model()
#     except Exception as e:
#         st.error(f"模型加载失败，请检查 {MODEL_PATH} 是否和 app.py 在同一目录下。\n\n错误详情: {e}")
#         st.stop()
#
#     model_obj     = bundle["model_obj"]
#     feature_names = bundle["feature_names"]   # 预测时必须严格按这个顺序
#
#     st.markdown("---")
#     st.subheader("请输入患者各项指标")
#
#     inputs = {}
#     cols = st.columns(2)
#     for i, fname in enumerate(feature_names):
#         lo, hi, default, step = INPUT_RANGES.get(fname, (0.0, 1000.0, 0.0, 1.0))
#         label = DISPLAY_LABELS.get(fname, fname)
#         with cols[i % 2]:
#             inputs[fname] = st.number_input(
#                 label, min_value=float(lo), max_value=float(hi),
#                 value=float(default), step=float(step), format="%.2f",
#             )
#
#     st.markdown("---")
#
#     if st.button("开始预测", type="primary", use_container_width=True):
#         X = np.array([[inputs[f] for f in feature_names]], dtype=float)
#
#         risk = float(predict_rsf_risk(model_obj, X)[0])
#
#         # 画图用：细密时间网格，画出平滑曲线（不受 model_obj 里只有6个采样点的限制）
#         plot_grid = np.linspace(0.5, max(MONTH_POINTS), 150)
#         t_grid, surv_mat = predict_rsf_survival_function(model_obj, X, t_grid=plot_grid)
#         surv = surv_mat[0]
#
#         # 表格用：精确取6个标准时间点的数值
#         _, surv_table_mat = predict_rsf_survival_function(
#             model_obj, X, t_grid=np.array(MONTH_POINTS, dtype=float))
#         surv_table = surv_table_mat[0]
#
#         is_high = risk >= RISK_CUTOFF
#
#         st.subheader("预测结果")
#         c1, c2 = st.columns(2)
#         with c1:
#             st.metric("风险分数 (Risk score)", f"{risk:.3f}")
#         with c2:
#             if is_high:
#                 st.error(f"⚠️ 高风险组\n\n(截断值 = {RISK_CUTOFF})")
#             else:
#                 st.success(f"✅ 低风险组\n\n(截断值 = {RISK_CUTOFF})")
#
#         st.subheader("个体化生存概率曲线")
#         fig, ax = plt.subplots(figsize=(7, 4))
#         color = "#C0392B" if is_high else "#2980B9"
#         ax.step(t_grid, surv, where="post", color=color, linewidth=2.2)
#         ax.fill_between(t_grid, surv, step="post", color=color, alpha=0.08)
#         ax.set_xlabel("随访时间 Follow-up time (months)")
#         ax.set_ylabel("预测生存概率 Predicted survival probability")
#         ax.set_ylim(0, 1.03)
#         ax.set_xlim(0, t_grid.max())
#         ax.grid(alpha=0.3)
#         st.pyplot(fig)
#
#         st.subheader("关键时间点生存概率")
#         rows = []
#         for mo, sp in zip(MONTH_POINTS, surv_table):
#             rows.append((f"{mo} 个月", f"{sp*100:.1f}%"))
#         st.table({"随访时间点": [r[0] for r in rows],
#                   "预测生存概率": [r[1] for r in rows]})
#
#     st.markdown("---")
#     st.caption(
#         "⚠️ 本工具基于回顾性队列训练的机器学习模型，仅供科研与临床参考，"
#         "不能替代医生的综合判断，结果应结合患者具体临床情况解读。"
#     )
#
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
HCM 患者心血管死亡风险 —— 个体化预测网页计算器（中英双语版）
HCM Cardiovascular Death Risk — Individualized Prediction Web Calculator (Bilingual)
基于随机生存森林 (Random Survival Forest) 模型

运行方式（本地测试）:
    streamlit run app.py
"""

import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# 配置 / Configuration
# =============================================================================
MODEL_PATH   = "RandomForest_final.pkl"
RISK_CUTOFF  = 2.324
MONTH_POINTS = [12, 24, 36, 48, 60, 72]

# =============================================================================
# 翻译字典 / Translation Dictionary
# key: translation key
# value: {"zh": Chinese text, "en": English text}
# =============================================================================
TRANS = {
    # --- 页面标题 / Page titles ---
    "page_title":  {"zh": "HCM心血管死亡风险预测",
                    "en": "HCM Cardiovascular Death Risk Prediction"},
    "title":       {"zh": "肥厚型心肌病（HCM）心血管死亡风险预测",
                    "en": "Hypertrophic Cardiomyopathy (HCM) Cardiovascular Death Risk Prediction"},
    "caption":     {"zh": "基于随机生存森林 (Random Survival Forest) 模型的个体化预测工具",
                    "en": "Individualized prediction tool based on Random Survival Forest model"},

    # --- 语言切换按钮 / Language toggle ---
    "lang_label":  {"zh": "🌐 切换语言 / Switch Language",
                    "en": "🌐 切换语言 / Switch Language"},
    "switch_to":   {"zh": "Switch to English",
                    "en": "切换为中文"},

    # --- 表单区 / Form section ---
    "section_input":  {"zh": "请输入患者各项指标",
                       "en": "Enter Patient Parameters"},
    "btn_predict":    {"zh": "开始预测",
                       "en": "Run Prediction"},

    # --- 结果区 / Results section ---
    "section_result": {"zh": "预测结果",
                       "en": "Prediction Results"},
    "risk_score":     {"zh": "风险分数 (Risk score)",
                       "en": "Risk Score"},
    "high_risk":      {"zh": f"⚠️ 高风险组\n\n(截断值 = {RISK_CUTOFF})",
                       "en": f"⚠️ High Risk Group\n\n(Cutoff = {RISK_CUTOFF})"},
    "low_risk":       {"zh": f"✅ 低风险组\n\n(截断值 = {RISK_CUTOFF})",
                       "en": f"✅ Low Risk Group\n\n(Cutoff = {RISK_CUTOFF})"},

    # --- 生存曲线图 / Survival curve ---
    "section_curve":  {"zh": "个体化生存概率曲线",
                       "en": "Individualized Survival Probability Curve"},
    "xlabel":         {"zh": "随访时间 Follow-up time (months)",
                       "en": "Follow-up Time (months)"},
    "ylabel":         {"zh": "预测生存概率 Predicted Survival Probability",
                       "en": "Predicted Survival Probability"},

    # --- 关键时间点表格 / Key time-point table ---
    "section_table":  {"zh": "关键时间点生存概率",
                       "en": "Survival Probability at Key Time Points"},
    "col_time":       {"zh": "随访时间点",
                       "en": "Follow-up Time"},
    "col_surv":       {"zh": "预测生存概率",
                       "en": "Predicted Survival Probability"},
    "month_unit":     {"zh": "个月",
                       "en": "months"},

    # --- 底部免责声明 / Disclaimer ---
    "disclaimer":     {"zh": ("⚠️ 本工具基于回顾性队列训练的机器学习模型，"
                              "仅供科研与临床参考，不能替代医生的综合判断，"
                              "结果应结合患者具体临床情况解读。"),
                       "en": ("⚠️ This tool is based on a machine-learning model trained on a "
                              "retrospective cohort. It is intended for research and clinical "
                              "reference only and does not replace comprehensive physician judgment. "
                              "Results should be interpreted in the context of each patient's "
                              "specific clinical situation.")},

    # --- 模型加载错误 / Model load error ---
    "model_error":    {"zh": ("模型加载失败，请检查 {path} 是否和 app.py 在同一目录下。\n\n"
                              "错误详情: {err}"),
                       "en": ("Failed to load model. Please check that {path} is in the same "
                              "directory as app.py.\n\nError details: {err}")},
}

# --- 特征变量双语标签 / Bilingual feature labels ---
DISPLAY_LABELS = {
    "age":          {"zh": "年龄 Age (y)",
                     "en": "Age (y)"},
    "BMI":          {"zh": "体重指数 BMI (kg/m²)",
                     "en": "BMI (kg/m²)"},
    "LVOT":         {"zh": "左室流出道压差 LVOT (mmHg)",
                     "en": "LVOT Gradient (mmHg)"},
    "LA_dimension": {"zh": "左房内径 LA dimension (mm)",
                     "en": "LA Dimension (mm)"},
    "LV_diameter":  {"zh": "左室内径 LV diameter (mm)",
                     "en": "LV Diameter (mm)"},
    "MWT":          {"zh": "最大室壁厚度 MWT (mm)",
                     "en": "Max Wall Thickness — MWT (mm)"},
    "LVEF":         {"zh": "左室射血分数 LVEF (%)",
                     "en": "LVEF (%)"},
    "CI":           {"zh": "心脏指数 CI (L/min/m²)",
                     "en": "Cardiac Index — CI (L/min/m²)"},
    "LVEDVi":       {"zh": "左室舒张末容积指数 LVEDVi (mL/m²)",
                     "en": "LVEDVi (mL/m²)"},
    "LVM_index":    {"zh": "左室质量指数 LVM index (g/m²)",
                     "en": "LV Mass Index (g/m²)"},
    "LGE_extent":   {"zh": "延迟强化范围 LGE extent (%)",
                     "en": "LGE Extent (%)"},
    "GLS":          {"zh": "整体纵向应变 GLS (%)",
                     "en": "Global Longitudinal Strain — GLS (%)"},
    "GCS":          {"zh": "整体周向应变 GCS (%)",
                     "en": "Global Circumferential Strain — GCS (%)"},
}

INPUT_RANGES = {
    "age":          (1.0,   100.0, 50.0,  1.0),
    "BMI":          (10.0,  60.0,  24.0,  0.1),
    "LVOT":         (0.0,   150.0, 20.0,  1.0),
    "LA_dimension": (20.0,  80.0,  40.0,  1.0),
    "LV_diameter":  (20.0,  80.0,  45.0,  1.0),
    "MWT":          (5.0,   50.0,  18.0,  1.0),
    "LVEF":         (5.0,   90.0,  65.0,  1.0),
    "CI":           (0.5,   8.0,   3.0,   0.1),
    "LVEDVi":       (10.0,  250.0, 70.0,  1.0),
    "LVM_index":    (20.0,  400.0, 100.0, 1.0),
    "LGE_extent":   (0.0,   80.0,  5.0,   1.0),
    "GLS":          (-35.0, 5.0,   -15.0, 1.0),
    "GCS":          (-40.0, 5.0,   -20.0, 1.0),
}


# =============================================================================
# 便捷翻译函数 / Helper translation function
# =============================================================================
def t(key: str, lang: str, **kwargs) -> str:
    """Return translated string for key in given lang ('zh' or 'en')."""
    text = TRANS[key][lang]
    return text.format(**kwargs) if kwargs else text


# =============================================================================
# 模型加载与预测函数
# =============================================================================
@st.cache_resource
def load_model():
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
    avg = np.mean(mats, axis=0)
    return t_grid, avg


# =============================================================================
# 主页面
# =============================================================================
def main():
    # ---- 语言状态初始化 / Language state init ----
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    lang = st.session_state["lang"]

    st.set_page_config(page_title=t("page_title", lang), layout="centered")

    # ---- 语言切换按钮（顶部右对齐） / Language toggle button (top right) ----
    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button(t("switch_to", lang), key="lang_btn"):
            st.session_state["lang"] = "en" if lang == "zh" else "zh"
            st.rerun()

    # 重新读取可能已切换的语言
    lang = st.session_state["lang"]

    st.title(t("title", lang))
    st.caption(t("caption", lang))

    # ---- 模型加载 / Load model ----
    try:
        bundle = load_model()
    except Exception as e:
        st.error(t("model_error", lang, path=MODEL_PATH, err=e))
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

        plot_grid = np.linspace(0.5, max(MONTH_POINTS), 150)
        t_grid, surv_mat = predict_rsf_survival_function(model_obj, X, t_grid=plot_grid)
        surv = surv_mat[0]

        _, surv_table_mat = predict_rsf_survival_function(
            model_obj, X, t_grid=np.array(MONTH_POINTS, dtype=float))
        surv_table = surv_table_mat[0]

        is_high = risk >= RISK_CUTOFF

        # ---- 风险分组结果 / Risk group result ----
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

        # 使用支持 Unicode 的字体，避免中文方块
        mpl.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS",
                                        "Microsoft YaHei", "sans-serif"]
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
        rows = {
            t("col_time", lang):  [f"{mo} {unit}" for mo in MONTH_POINTS],
            t("col_surv", lang):  [f"{sp * 100:.1f}%" for sp in surv_table],
        }
        st.table(rows)

    st.markdown("---")
    st.caption(t("disclaimer", lang))


if __name__ == "__main__":
    main()