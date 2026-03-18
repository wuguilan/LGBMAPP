import streamlit as st
import pandas as pd
import joblib
import lightgbm  # 必须导入以确保joblib能正确加载LGBM模型
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
@st.cache_resource  # 使用缓存，避免每次都重新加载模型
def load_model(path):
    """加载 .joblib 格式的模型"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"错误：模型文件 '{path}' 未找到。")
        st.error(f"请确保 '{path}' 文件与您的Streamlit应用在同一个目录下。")
        return None
    except Exception as e:
        st.error(f"加载模型时发生未知错误: {e}")
        return None

# 加载您训练好的LightGBM模型
lgbm_model = load_model('LightGBM.joblib')

# --- 特征定义 ---
# 定义模型训练时使用的完整特征列
FEATURE_COLUMNS = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure", "albumin_max",
    "creatinine_max", "inr_min", "pt_min", "alt_max", "fresh_frozen_plasma_input",
    "platelets_input", "rbw_input", "vasopressin", "sedative", "cvc"
]

# 将特征分为数值型和二元（是/否）型
NUMERIC_FEATURES = [
    "albumin_max", "creatinine_max", "inr_min", "pt_min", "alt_max",
    "fresh_frozen_plasma_input", "platelets_input", "rbw_input"
]
BINARY_FEATURES = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure",
    "vasopressin", "sedative", "cvc"
]

# 为数值特征设置默认值
DEFAULT_VALUES = {
    "albumin_max": 40.0,
    "creatinine_max": 80.0,
    "inr_min": 1.0,
    "pt_min": 12.0,
    "alt_max": 25.0,
    "fresh_frozen_plasma_input": 0.0,
    "platelets_input": 0.0,
    "rbw_input": 0.0
}

# --- 创建SHAP解释器（缓存）---
@st.cache_resource
def get_shap_explainer(model):
    """创建并缓存SHAP解释器"""
    if model is not None:
        return shap.TreeExplainer(model)
    return None

# --- 页面标题 ---
st.title("🩸 基于LightGBM的VTE事件风险预测系统(Alfafa-sepsis-vte)")
st.markdown("---")

# --- 用户输入界面 ---
if lgbm_model:
    # 初始化SHAP解释器
    explainer = get_shap_explainer(lgbm_model)
    
    with st.expander("点击此处输入/修改患者指标", expanded=True):
        input_data = {}

        with st.form("vte_input_form"):
            st.subheader("数值指标")
            num_cols = st.columns(4)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with num_cols[i % 4]:
                    input_data[feature] = st.number_input(
                        label=feature,
                        step=1.0,
                        format="%.2f",
                        value=DEFAULT_VALUES.get(feature, 0.0)
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("二元指标 (是/否)")
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    value = st.radio(
                        label=feature,
                        options=['否', '是'],
                        key=f"radio_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == '是' else 0

            submitted = st.form_submit_button("执行VTE风险预测")

    # --- 预测和结果展示 ---
    if submitted:
        st.header("📊 预测结果与个体化解释")
        
        # 创建输入DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # 预测概率
        prediction_proba = lgbm_model.predict_proba(input_df)[:, 1][0]
        
        # 风险分层
        risk_level, risk_color = "", ""
        if prediction_proba <= 0.0078:
            risk_level, risk_color = "低风险", "green"
        elif 0.0078 < prediction_proba <= 0.0294:
            risk_level, risk_color = "中风险", "orange"
        else:
            risk_level, risk_color = "高风险", "red"
        
        # 显示预测结果
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="VTE事件预测概率", value=f"{prediction_proba:.4%}")
        with col2:
            st.markdown(f"### 风险等级: <font color='{risk_color}'>**{risk_level}**</font>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- SHAP可视化（修复版本）---
        st.subheader("个体化预测归因 (SHAP Waterfall)")
        st.markdown(
            "下图解释了每个特征如何将预测概率从基线值推向最终的输出值。"
            "**红色**的特征是增加风险的因素，**蓝色**的特征是降低风险的因素。"
        )
        
        try:
            # 计算SHAP值 - 兼容新版shap
            if hasattr(explainer, 'expected_value'):
                # 对于二分类，expected_value可能是一个列表
                if isinstance(explainer.expected_value, list):
                    expected_value = explainer.expected_value[1]  # 使用正类的期望值
                else:
                    expected_value = explainer.expected_value
                
                # 计算SHAP值
                shap_values = explainer.shap_values(input_df)
                
                # 对于二分类，shap_values是一个列表，取正类的SHAP值
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1][0]  # 正类的SHAP值
                else:
                    shap_values_for_plot = shap_values[0]
                
                # 创建Explanation对象
                shap_exp = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=input_df.columns.tolist()
                )
                
                # 绘制waterfall图
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap_exp, show=False)
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"SHAP可视化过程中出现错误: {str(e)}")
            st.info("尝试使用备选的可视化方法...")
            
            try:
                # 备选方案：使用简单的条形图
                shap_values = explainer.shap_values(input_df)
                if isinstance(shap_values, list):
                    shap_values_to_plot = shap_values[1][0]  # 取正类的SHAP值
                else:
                    shap_values_to_plot = shap_values[0]
                
                # 创建简单的条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importance = pd.DataFrame({
                    'feature': input_df.columns,
                    'shap_value': shap_values_to_plot
                }).sort_values('shap_value', key=abs, ascending=False).head(10)
                
                colors = ['red' if x > 0 else 'blue' for x in feature_importance['shap_value']]
                ax.barh(feature_importance['feature'], feature_importance['shap_value'], color=colors)
                ax.set_xlabel('SHAP value (对预测的影响)')
                ax.set_title('特征对预测的贡献（红色：增加风险，蓝色：降低风险）')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e2:
                st.error(f"备选可视化也失败了: {str(e2)}")
                st.write("无法生成SHAP图，但预测结果仍然有效。")
        
        # 显示输入数据详情
        with st.expander("查看本次输入的详细信息"):
            styled_df = input_df.style.highlight_max(axis=1)
            st.dataframe(styled_df)
            
            # 显示概率详情
            proba_df = pd.DataFrame({
                '类别': ['无VTE事件', '有VTE事件'],
                '概率': [f"{1-prediction_proba:.4%}", f"{prediction_proba:.4%}"]
            })
            st.write("预测概率详情:")
            st.dataframe(proba_df)

else:
    st.warning("模型未能加载，应用无法运行。请检查 'LightGBM.joblib' 文件是否存在。")

# --- 侧边栏信息 ---
with st.sidebar:
    st.header("📋 关于本系统")
    st.markdown("""
    **VTE风险预测系统** (Alfafa-sepsis-vte)
    
    - **模型**: LightGBM 二分类
    - **预测目标**: VTE事件风险
    - **特征数量**: 15个临床指标
    
    **风险分层标准**:
    - 🟢 **低风险**: ≤ 0.78%
    - 🟡 **中风险**: 0.78% - 2.94%
    - 🔴 **高风险**: > 2.94%
    """)
    
    st.warning("""
    **临床免责声明**
    本工具仅供参考，不能替代专业医疗判断。
    所有临床决策都应结合患者具体情况。
    """)
