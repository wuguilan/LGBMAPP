import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import sys
import os
import pickle

# 设置页面配置（必须是第一个streamlit命令）
st.set_page_config(
    page_title="VTE风险预测系统",
    page_icon="🩸",
    layout="wide"
)

# 页面标题
st.title("🩸 基于LightGBM的VTE事件风险预测系统")
st.markdown("---")

# 侧边栏信息
with st.sidebar:
    st.header("📋 系统信息")
    st.write(f"Python版本: {sys.version.split()[0]}")
    st.write(f"NumPy版本: {np.__version__}")
    st.write(f"Joblib版本: {joblib.__version__}")
    st.write(f"LightGBM版本: {lgb.__version__}")
    
    st.header("📊 风险分层标准")
    st.markdown("""
    - 🟢 **低风险**: ≤ 0.78%
    - 🟡 **中风险**: 0.78% - 2.94%
    - 🔴 **高风险**: > 2.94%
    """)
    
    st.warning("""
    **临床免责声明**
    本工具仅供参考，不能替代专业医疗判断。
    """)

# 特征定义
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

# 特征显示名称（用于界面）
FEATURE_DISPLAY_NAMES = {
    "vte_history": "VTE病史",
    "cancer": "癌症",
    "respiratory_failure": "呼吸衰竭",
    "heart_failure": "心力衰竭",
    "albumin_max": "白蛋白最大值 (g/dL)",
    "creatinine_max": "肌酐最大值 (mg/dL)",
    "inr_min": "INR最小值",
    "pt_min": "PT最小值 (秒)",
    "alt_max": "ALT最大值 (U/L)",
    "fresh_frozen_plasma_input": "新鲜冰冻血浆输入",
    "platelets_input": "血小板输入",
    "rbw_input": "红细胞输入",
    "vasopressin": "血管加压素",
    "sedative": "镇静剂",
    "cvc": "中心静脉导管"
}

# 默认值
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

# 模型加载函数
@st.cache_resource
def load_model_with_compatibility(path):
    """
    兼容不同numpy版本的模型加载函数
    """
    if not os.path.exists(path):
        st.error(f"❌ 模型文件不存在: {path}")
        st.info(f"当前目录内容: {os.listdir('.')}")
        return None
    
    try:
        # 方法1: 直接使用joblib加载
        st.info("尝试使用joblib加载模型...")
        model = joblib.load(path)
        st.success("✅ joblib加载成功")
        return model
    except Exception as e1:
        st.warning(f"joblib加载失败: {e1}")
        
        try:
            # 方法2: 使用pickle加载
            st.info("尝试使用pickle加载模型...")
            with open(path, 'rb') as f:
                model = pickle.load(f)
            st.success("✅ pickle加载成功")
            return model
        except Exception as e2:
            st.warning(f"pickle加载失败: {e2}")
            
            try:
                # 方法3: 使用joblib + 兼容模式
                st.info("尝试使用兼容模式加载...")
                
                # 创建一个兼容的numpy命名空间
                class CompatibleNP:
                    pass
                
                # 添加可能需要的numpy模块引用
                if hasattr(np, 'core'):
                    np.core = np.core
                if hasattr(np, '_core'):
                    np._core = np._core
                
                # 使用自定义的加载器
                model = joblib.load(path)
                st.success("✅ 兼容模式加载成功")
                return model
                
            except Exception as e3:
                st.error(f"所有加载方法都失败: {e3}")
                
                # 显示文件信息用于调试
                file_size = os.path.getsize(path)
                st.info(f"文件大小: {file_size} bytes")
                
                # 尝试读取文件头部
                try:
                    with open(path, 'rb') as f:
                        header = f.read(100)
                    st.info(f"文件头部 (hex): {header[:50].hex()}")
                except:
                    pass
                
                return None

# 加载模型
MODEL_PATH = 'LightGBM.joblib'
st.sidebar.write(f"模型文件: {MODEL_PATH}")
st.sidebar.write(f"文件存在: {os.path.exists(MODEL_PATH)}")

model = load_model_with_compatibility(MODEL_PATH)

# 主界面
if model is not None:
    st.success("✅ 模型加载成功！")
    
    # 显示模型信息
    with st.expander("模型信息", expanded=False):
        st.write(f"模型类型: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            st.write(f"特征数量: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            st.write(f"类别: {model.classes_}")
    
    # 用户输入表单
    with st.form("prediction_form"):
        st.subheader("📝 患者临床指标输入")
        
        # 创建三列布局
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        # 分配特征到各列
        with col1:
            st.markdown("**基本特征**")
            input_data["vte_history"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["vte_history"], 
                ['无', '有'], 
                horizontal=True,
                key="vte_history"
            ) == '有' else 0
            
            input_data["cancer"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["cancer"], 
                ['无', '有'], 
                horizontal=True,
                key="cancer"
            ) == '有' else 0
            
            input_data["respiratory_failure"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["respiratory_failure"], 
                ['无', '有'], 
                horizontal=True,
                key="respiratory_failure"
            ) == '有' else 0
            
            input_data["heart_failure"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["heart_failure"], 
                ['无', '有'], 
                horizontal=True,
                key="heart_failure"
            ) == '有' else 0
        
        with col2:
            st.markdown("**实验室指标**")
            input_data["albumin_max"] = st.number_input(
                FEATURE_DISPLAY_NAMES["albumin_max"],
                min_value=0.0, max_value=10.0, 
                value=DEFAULT_VALUES["albumin_max"],
                step=0.1, format="%.2f"
            )
            
            input_data["creatinine_max"] = st.number_input(
                FEATURE_DISPLAY_NAMES["creatinine_max"],
                min_value=0.0, max_value=20.0,
                value=DEFAULT_VALUES["creatinine_max"],
                step=0.1, format="%.2f"
            )
            
            input_data["inr_min"] = st.number_input(
                FEATURE_DISPLAY_NAMES["inr_min"],
                min_value=0.5, max_value=5.0,
                value=DEFAULT_VALUES["inr_min"],
                step=0.1, format="%.2f"
            )
            
            input_data["pt_min"] = st.number_input(
                FEATURE_DISPLAY_NAMES["pt_min"],
                min_value=5.0, max_value=50.0,
                value=DEFAULT_VALUES["pt_min"],
                step=0.1, format="%.2f"
            )
            
            input_data["alt_max"] = st.number_input(
                FEATURE_DISPLAY_NAMES["alt_max"],
                min_value=0.0, max_value=1000.0,
                value=DEFAULT_VALUES["alt_max"],
                step=1.0, format="%.1f"
            )
        
        with col3:
            st.markdown("**治疗相关**")
            input_data["fresh_frozen_plasma_input"] = st.number_input(
                FEATURE_DISPLAY_NAMES["fresh_frozen_plasma_input"],
                min_value=0.0, max_value=20.0,
                value=DEFAULT_VALUES["fresh_frozen_plasma_input"],
                step=0.1, format="%.2f"
            )
            
            input_data["platelets_input"] = st.number_input(
                FEATURE_DISPLAY_NAMES["platelets_input"],
                min_value=0.0, max_value=20.0,
                value=DEFAULT_VALUES["platelets_input"],
                step=0.1, format="%.2f"
            )
            
            input_data["rbw_input"] = st.number_input(
                FEATURE_DISPLAY_NAMES["rbw_input"],
                min_value=0.0, max_value=20.0,
                value=DEFAULT_VALUES["rbw_input"],
                step=0.1, format="%.2f"
            )
            
            input_data["vasopressin"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["vasopressin"], 
                ['无', '有'], 
                horizontal=True,
                key="vasopressin"
            ) == '有' else 0
            
            input_data["sedative"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["sedative"], 
                ['无', '有'], 
                horizontal=True,
                key="sedative"
            ) == '有' else 0
            
            input_data["cvc"] = 1 if st.radio(
                FEATURE_DISPLAY_NAMES["cvc"], 
                ['无', '有'], 
                horizontal=True,
                key="cvc"
            ) == '有' else 0
        
        # 提交按钮
        submitted = st.form_submit_button("🔮 预测VTE风险", type="primary", use_container_width=True)
    
    # 预测结果
    if submitted:
        st.markdown("---")
        st.subheader("📊 预测结果")
        
        # 创建DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        try:
            # 进行预测
            prediction_proba = model.predict_proba(input_df)[0][1]
            
            # 风险分层
            if prediction_proba <= 0.0078:
                risk_level = "低风险"
                risk_color = "green"
                risk_emoji = "🟢"
            elif prediction_proba <= 0.0294:
                risk_level = "中风险"
                risk_color = "orange"
                risk_emoji = "🟡"
            else:
                risk_level = "高风险"
                risk_color = "red"
                risk_emoji = "🔴"
            
            # 显示主要结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VTE预测概率", f"{prediction_proba:.2%}")
            with col2:
                st.metric("风险等级", f"{risk_emoji} {risk_level}")
            with col3:
                st.metric("无VTE概率", f"{(1-prediction_proba):.2%}")
            
            # 风险提示框
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}20; 
                        border-left: 5px solid {risk_color}; margin: 20px 0;'>
                <h3 style='color: {risk_color}; margin: 0;'>{risk_emoji} {risk_level}</h3>
                <p style='margin: 10px 0 0 0; font-size: 18px;'>
                    VTE发生概率: {prediction_proba:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                st.subheader("📈 特征重要性分析")
                
                importance_df = pd.DataFrame({
                    '特征': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLUMNS],
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                bars = ax.barh(importance_df['特征'], importance_df['重要性'])
                ax.set_xlabel('特征重要性')
                ax.set_title('LightGBM特征重要性排名')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars, importance_df['重要性'])):
                    ax.text(val, i, f' {val:.0f}', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # 显示输入数据
            with st.expander("查看输入数据详情"):
                display_df = input_df.copy()
                display_df.columns = [FEATURE_DISPLAY_NAMES.get(c, c) for c in display_df.columns]
                st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.exception(e)

else:
    st.error("❌ 无法加载模型，请检查以下问题：")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**可能的原因：**")
        st.write("1. 模型文件不存在")
        st.write("2. 模型文件损坏")
        st.write("3. NumPy版本不兼容")
        st.write("4. 文件权限问题")
    
    with col2:
        st.write("**建议解决方案：**")
        st.write("1. 确认 'LightGBM.joblib' 文件已上传")
        st.write("2. 在本地重新保存模型文件")
        st.write("3. 检查文件大小是否正常")
    
    # 调试信息
    with st.expander("🔧 调试信息"):
        st.write(f"当前工作目录: {os.getcwd()}")
        st.write(f"目录内容: {os.listdir('.')}")
        
        if os.path.exists(MODEL_PATH):
            st.write(f"模型文件大小: {os.path.getsize(MODEL_PATH)} 字节")
            st.write(f"模型文件权限: {oct(os.stat(MODEL_PATH).st_mode)[-3:]}")
        else:
            st.write("模型文件不存在")
