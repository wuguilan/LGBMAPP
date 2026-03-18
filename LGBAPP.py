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
@st.cache_resource
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
    # 创建SHAP解释器（延迟初始化，只在需要时创建）
    @st.cache_resource
    def get_explainer(model):
        try:
            return shap.TreeExplainer(model)
        except Exception as e:
            st.warning(f"SHAP解释器初始化失败: {e}")
            return None
    
    explainer = get_explainer(lgbm_model)
    
    with st.expander("点击此处输入/修改患者指标", expanded=True):
        input_data = {}

        with st.form("vte_input_form"):
            st.subheader("数值指标")
            num_cols = st.columns(4)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with num_cols[i % 4]:
                    input_data[feature] = st.number_input(
                        label=feature,
                        step=0.1,
                        format="%.2f",
                        value=float(DEFAULT_VALUES.get(feature, 0.0))
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

            submitted = st.form_submit_button("执行VTE风险预测", type="primary")

    # --- 预测和结果展示 ---
    if submitted:
        st.header("📊 预测结果与个体化解释")

        # 创建输入DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # 显示输入数据
        with st.expander("查看输入的详细数据"):
            st.dataframe(input_df)

        try:
            # 预测概率
            prediction_proba = lgbm_model.predict_proba(input_df)[:, 1][0]

            # 风险分层
            if prediction_proba <= 0.0078:
                risk_level, risk_color = "低风险", "green"
                risk_emoji = "🟢"
            elif 0.0078 < prediction_proba <= 0.0294:
                risk_level, risk_color = "中风险", "orange"
                risk_emoji = "🟡"
            else:
                risk_level, risk_color = "高风险", "red"
                risk_emoji = "🔴"

            # 显示预测结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="VTE事件预测概率", value=f"{prediction_proba:.2%}")
            with col2:
                st.metric(label="风险等级", value=f"{risk_emoji} {risk_level}")
            with col3:
                st.metric(label="置信度", value=f"{max(prediction_proba, 1-prediction_proba):.2%}")

            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}20; border-left: 5px solid {risk_color};'>
                <h3 style='color: {risk_color}; margin: 0;'>预测结果: {risk_emoji} {risk_level}</h3>
                <p style='margin: 10px 0 0 0;'>VTE发生概率: {prediction_proba:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # SHAP 可视化
            if explainer is not None:
                st.subheader("📈 个体化预测归因分析 (SHAP)")
                st.markdown("""
                **红色**：增加风险的因素  
                **蓝色**：降低风险的因素  
                *横坐标表示该特征对预测结果的影响程度*
                """)
                
                try:
                    # 计算SHAP值
                    shap_values = explainer.shap_values(input_df)
                    
                    # 处理二分类的SHAP值
                    if isinstance(shap_values, list):
                        # 对于二分类，shap_values是一个列表，取正类的SHAP值
                        shap_values_for_plot = shap_values[1][0]
                        
                        # 获取期望值
                        if isinstance(explainer.expected_value, list):
                            expected_value = explainer.expected_value[1]
                        else:
                            expected_value = explainer.expected_value
                    else:
                        shap_values_for_plot = shap_values[0]
                        expected_value = explainer.expected_value
                    
                    # 创建Explanation对象
                    shap_exp = shap.Explanation(
                        values=shap_values_for_plot,
                        base_values=expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=input_df.columns.tolist()
                    )
                    
                    # 绘制waterfall图
                    fig, ax = plt.subplots(figsize=(12, 6))
                    shap.waterfall_plot(shap_exp, show=False, max_display=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"SHAP waterfall图生成失败: {e}")
                    
                    # 备选方案：使用条形图
                    try:
                        st.subheader("备选方案：特征重要性条形图")
                        
                        shap_values = explainer.shap_values(input_df)
                        if isinstance(shap_values, list):
                            shap_values_to_plot = shap_values[1][0]
                        else:
                            shap_values_to_plot = shap_values[0]
                        
                        # 创建特征重要性DataFrame
                        feature_imp = pd.DataFrame({
                            'feature': input_df.columns,
                            'shap_value': shap_values_to_plot
                        }).sort_values('shap_value', key=abs, ascending=True)
                        
                        # 绘制条形图
                        fig, ax = plt.subplots(figsize=(10, 8))
                        colors = ['red' if x > 0 else 'blue' for x in feature_imp['shap_value']]
                        ax.barh(feature_imp['feature'], feature_imp['shap_value'], color=colors)
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        ax.set_xlabel('SHAP值 (对预测的影响)')
                        ax.set_title('特征对预测的贡献')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e2:
                        st.error(f"备选可视化也失败了: {e2}")
            
            else:
                st.info("SHAP解释器未初始化，无法生成归因分析图")
                
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.exception(e)

else:
    st.warning("⚠️ 模型未能加载，应用无法运行。请检查 'LightGBM.joblib' 文件是否存在。")
    
    # 显示当前目录内容供调试
    if st.checkbox("显示调试信息"):
        import os
        st.write("当前目录内容:")
        st.write(os.listdir('.'))

