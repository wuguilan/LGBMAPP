import streamlit as st
import pandas as pd
import joblib
import lightgbm  # 确保 joblib 能加载 LGBM
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- 页面基础配置 ---
st.set_page_config(
    page_title="VTE风险预测与解释系统(Alfafa-sepsis-vte)",
    page_icon="🩸",
    layout="wide"
)

# --- 模型加载 ---
@st.cache_resource
def load_model(path):
    """加载 .joblib 格式的 LightGBM 模型"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"错误：模型文件 '{path}' 未找到。")
        st.error(f"请确保 '{path}' 文件与Streamlit应用在同目录下。")
        return None
    except Exception as e:
        st.error(f"加载模型时发生未知错误: {e}")
        return None

# 加载模型
lgbm_model = load_model("LightGBM.joblib")

# --- 特征定义 ---
FEATURE_COLUMNS = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure", "albumin_max",
    "creatinine_max", "inr_min", "pt_min", "alt_max", "fresh_frozen_plasma_input",
    "platelets_input", "rbw_input", "vasopressin", "sedative", "cvc"
]

NUMERIC_FEATURES = [
    "albumin_max", "creatinine_max", "inr_min", "pt_min", "alt_max",
    "fresh_frozen_plasma_input", "platelets_input", "rbw_input"
]

BINARY_FEATURES = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure",
    "vasopressin", "sedative", "cvc"
]

# 特征英文名称（用于SHAP图）
FEATURE_NAMES_EN = {
    "vte_history": "VTE History",
    "cancer": "Cancer",
    "respiratory_failure": "Resp Failure",
    "heart_failure": "Heart Failure",
    "albumin_max": "Albumin",
    "creatinine_max": "Creatinine",
    "inr_min": "INR",
    "pt_min": "PT",
    "alt_max": "ALT",
    "fresh_frozen_plasma_input": "FFP",
    "platelets_input": "Platelets",
    "rbw_input": "RBC",
    "vasopressin": "Vasopressin",
    "sedative": "Sedative",
    "cvc": "CVC"
}

# 默认值设置
DEFAULT_VALUES = {
    "albumin_max": 2.4,
    "creatinine_max": 1.6,
    "inr_min": 1.1,
    "pt_min": 12.7,
    "alt_max": 46.0,
    "fresh_frozen_plasma_input": 0.0,
    "platelets_input": 0.0,
    "rbw_input": 0.0
}

# --- 页面标题 ---
st.title("🩸 基于LightGBM的VTE事件风险预测系统(Alfafa-sepsis-vte)")
st.markdown("---")

# --- 用户输入界面 ---
if lgbm_model:
    with st.expander("点击此处输入患者指标", expanded=True):
        input_data = {}
        
        with st.form("input_form"):
            st.subheader("📊 数值型指标")
            
            # 3列布局
            cols = st.columns(3)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with cols[i % 3]:
                    # 为每个特征添加中文说明和单位
                    if feature == 'albumin_max':
                        label = "白蛋白 (g/dL)"
                    elif feature == 'creatinine_max':
                        label = "肌酐(mg/dL)"
                    elif feature == 'inr_min':
                        label = "INR"
                    elif feature == 'pt_min':
                        label = "PT(秒)"
                    elif feature == 'alt_max':
                        label = "ALT (U/L)"
                    elif feature == 'fresh_frozen_plasma_input':
                        label = "新鲜冰冻血浆输入 (单位)"
                    elif feature == 'platelets_input':
                        label = "血小板输入 (单位)"
                    elif feature == 'rbw_input':
                        label = "红细胞输入 (单位)"
                    else:
                        label = FEATURE_NAMES_EN.get(feature, feature)
                    
                    input_data[feature] = st.number_input(
                        label=label,
                        min_value=0.0,
                        max_value=100.0 if feature in ['albumin_max', 'creatinine_max'] else 1000.0,
                        value=float(DEFAULT_VALUES.get(feature, 0.0)),
                        step=0.1,
                        format="%.1f",
                        key=f"num_{feature}"
                    )
            
            st.markdown("---")
            st.subheader("✅ 二分类指标 (是/否)")
            
            # 4列布局
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    # 为二分类特征添加中文说明
                    if feature == 'vte_history':
                        label = "VTE病史"
                    elif feature == 'cancer':
                        label = "癌症"
                    elif feature == 'respiratory_failure':
                        label = "呼吸衰竭"
                    elif feature == 'heart_failure':
                        label = "心力衰竭"
                    elif feature == 'vasopressin':
                        label = "血管加压素"
                    elif feature == 'sedative':
                        label = "镇静剂"
                    elif feature == 'cvc':
                        label = "中心静脉导管"
                    else:
                        label = FEATURE_NAMES_EN.get(feature, feature)
                    
                    value = st.radio(
                        label=label,
                        options=['否', '是'],
                        key=f"bin_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == '是' else 0
            
            submitted = st.form_submit_button("🔮 预测VTE风险", type="primary", use_container_width=True)

    # --- 预测和结果展示 ---
    if submitted:
        st.header("📈 预测结果与个体化解释")
        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        # 模型预测
        prediction_proba = lgbm_model.predict_proba(input_df)[:, 1][0]

        # 风险等级判断
        if prediction_proba <= 0.0078:
            risk_level, risk_color = "低风险", "green"
        elif prediction_proba <= 0.0294:
            risk_level, risk_color = "中风险", "orange"
        else:
            risk_level, risk_color = "高风险", "red"

        # 显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="VTE事件预测概率", value=f"{prediction_proba:.4%}")
        with col2:
            st.markdown(f"### 风险等级: <font color='{risk_color}'>**{risk_level}**</font>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("个体化预测归因 (SHAP Waterfall)")
        st.markdown(
            "下图解释了每个特征如何将预测概率从基线值（`base value`）推向最终输出。"
            "**红色**表示增加风险，**蓝色**表示降低风险。"
        )

        # SHAP解释
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(input_df)
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )
        st.pyplot(plt.gcf())

        with st.expander("查看本次输入的详细信息"):
            st.dataframe(input_df.style.highlight_max(axis=1))
else:
    st.warning("模型未能加载，应用无法运行。请检查 'LightGBM.joblib' 文件是否存在。")
