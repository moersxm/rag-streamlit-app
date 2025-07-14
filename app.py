import streamlit as st
import time
import os
import json
import requests
import datetime
import uuid
from rag_system import RAGSystem

# 文件上传与处理功能
class DocumentProcessor:
    def __init__(self, base_dir, manual_chunks_dir="manual_chunks", vector_db_dir="vector_db_manual"):
        self.base_dir = base_dir
        self.manual_chunks_dir = os.path.join(base_dir, manual_chunks_dir)
        self.vector_db_dir = os.path.join(base_dir, vector_db_dir)
        self.metadata_path = os.path.join(self.vector_db_dir, "metadata.json")
        
        # 确保目录存在
        os.makedirs(self.manual_chunks_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
    
    def load_metadata(self):
        """加载现有的元数据"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_metadata(self, metadata_list):
        """保存更新后的元数据"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

# 改进设置页面主题函数
def set_page_theme():
    # 在现有CSS基础上添加修复
    st.markdown("""
    <style>
        /* 全局样式 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
        
        body {
            font-family: 'Noto Sans SC', sans-serif;
            color: #1F2937;
            line-height: 1.6;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Noto Sans+SC', sans-serif;
            font-weight: 700;
        }
        
        /* 主标题 */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E40AF;
            margin-bottom: 1.5rem;
            text-align: center;
            padding: 1.5rem 0;
            border-bottom: 3px solid #E5E7EB;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeIn 1s ease;
        }
        
        /* 子标题 */
        .sub-header {
            font-size: 1.8rem;
            color: #1E40AF;
            margin: 1.8rem 0 1.2rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E5E7EB;
            font-weight: 600;
        }
        
        /* 回答容器 */
        .answer-container {
            background-color: #F9FAFB;
            padding: 2rem;
            border-radius: 0.8rem;
            margin: 1.5rem 0;
            border-left: 5px solid #3B82F6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            font-size: 1.05rem;
            line-height: 1.7;
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }
        
        .answer-container:hover {
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }
        
        .answer-header {
            font-weight: 600;
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #1E40AF;
            border-bottom: 1px solid #E5E7EB;
            padding-bottom: 0.8rem;
        }
        
        /* 信息文本 */
        .info-text {
            color: #4B5563;
            font-size: 1.05rem;
            line-height: 1.6;
            padding: 0.5rem 0;
        }
        
        .highlight-text {
            color: #1E40AF;
            font-weight: 600;
            background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%);
            padding: 0.1rem 0.4rem;
            border-radius: 0.3rem;
            color: white;
        }
        
        /* 来源容器 */
        .source-title {
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #1E40AF;
        }
        
        .source-container {
            border: 1px solid #E5E7EB;
            border-radius: 0.8rem;
            padding: 1.2rem;
            margin-bottom: 1rem;
            background-color: #FAFAFA;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        
        .source-container:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            border-color: #D1D5DB;
        }
        
        /* 指标卡片 */
        .metrics-card {
            background: linear-gradient(145deg, #F9FAFB, #F3F4F6);
            padding: 1.2rem;
            border-radius: 0.8rem;
            text-align: center;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
            border: 1px solid rgba(209, 213, 219, 0.5);
            height: 100%;
        }
        
        .metrics-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.15);
            border-color: #3B82F6;
        }
        
        .metrics-label {
            font-size: 1rem;
            color: #4B5563;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .metrics-value {
            font-size: 1.5rem;
            color: #1E40AF;
            font-weight: 700;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.05);
        }
        
        /* 页脚 */
        .footer {
            text-align: center;
            padding: 2.5rem 1rem;
            color: #6B7280;
            font-size: 0.9rem;
            margin-top: 3rem;
            border-top: 1px solid #E5E7EB;
            background-color: #F9FAFB;
            border-radius: 0.5rem;
            width: 100%;
            box-sizing: border-box;
        }
        
        .footer-content {
            display: inline-block;
            text-align: center;
            max-width: 100%;
        }
        
        .footer-flex {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .footer-item {
            margin: 0 10px;
            white-space: nowrap;
        }
        
        /* 自定义进度条样式 */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #2563EB, #60A5FA);
        }
        
        /* 自定义按钮样式 */
        .stButton button {
            background: linear-gradient(90deg, #1E40AF, #3B82F6);
            color: white;
            border-radius: 0.5rem;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .stButton button:hover {
            background: linear-gradient(90deg, #1E40AF, #2563EB);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        
        /* 自定义文本区域样式 */
        textarea {
            border-radius: 0.8rem !important;
            border: 2px solid #D1D5DB !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }
        
        textarea:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
            transform: translateY(-2px);
        }
        
        /* 历史记录样式 */
        .history-item {
            border-bottom: 1px solid #E5E7EB;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #F9FAFB;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }
        
        .history-item:hover {
            background-color: #F3F4F6;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .history-question {
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.5rem;
        }
        
        .history-answer {
            color: #4B5563;
        }
        
        /* 提示信息样式 */
        .notice-box {
            background-color: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            position: relative;
            overflow: hidden;
        }
        
        .notice-box::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(59, 130, 246, 0.05) 0%, rgba(37, 99, 235, 0) 70%);
            z-index: 0;
        }
        
        .notice-box p {
            position: relative;
            z-index: 1;
        }
        
        /* 问题显示样式 */
        .question-box {
            padding: 1rem;
            background-color: #F3F4F6;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #6B7280;
        }
        
        /* 免责声明样式 */
        .disclaimer-box {
            color: #B91C1C;
            font-weight: 500;
            margin-bottom: 1rem;
            padding: 0.8rem;
            background-color: #FEF2F2;
            border-radius: 0.5rem;
            border-left: 4px solid #DC2626;
            line-height: 1.5;
        }
        
        /* 自定义扩展器样式 */
        .streamlit-expanderHeader {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E40AF;
            background-color: #F3F4F6;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #E5E7EB;
        }
        
        /* 修复UI错位的额外CSS */
        .stButton > button {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* 修复图标对齐问题 */
        .emoji-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            vertical-align: middle;
            margin-right: 5px;
        }
        
        /* 修复页脚对齐问题 */
        .footer {
            text-align: center;
            padding: 2.5rem 1rem;
            color: #6B7280;
            font-size: 0.9rem;
            margin-top: 3rem;
            border-top: 1px solid #E5E7EB;
            background-color: #F9FAFB;
            border-radius: 0.5rem;
            width: 100%;
            box-sizing: border-box;
        }
        
        .footer-content {
            display: inline-block;
            text-align: center;
            max-width: 100%;
        }
        
        .footer-flex {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .footer-item {
            margin: 0 10px;
            white-space: nowrap;
        }
        
        /* 响应式调整改进 */
        @media (max-width: 768px) {
            .footer-flex {
                flex-direction: column;
                gap: 5px;
            }
            
            .footer-item {
                margin: 2px 0;
            }
            
            .divider {
                display: none;
            }
        }
        
        /* 添加动画效果 */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .answer-container {
            animation: slideIn 0.5s ease-in-out;
        }
        
        /* 添加滚动条样式 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F3F4F6;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #3B82F6;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #2563EB;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="政府采购和PPP项目智能问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用自定义主题
    set_page_theme()
    
    # 基础目录设置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vector_db_path = os.path.join(base_dir, "vector_db_manual")
    
    # 检查向量数据库目录是否存在
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path, exist_ok=True)
        st.warning(f"向量数据库目录不存在，已创建新目录: {vector_db_path}")
    
    # 检查index.faiss文件是否存在
    index_path = os.path.join(vector_db_path, "index.faiss")
    if not os.path.exists(index_path):
        st.warning("向量索引文件不存在，系统将尝试创建新索引。如果这是首次运行，请确保有足够的数据文件。")
        
        # 创建一个空的metadata.json文件以确保初始化不会失败
        metadata_path = os.path.join(vector_db_path, "metadata.json")
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            st.info("已创建空的元数据文件。请添加文档数据后重启应用。")
    
    # 检查并修复metadata.json中的文件路径
    try:
        metadata_path = os.path.join(vector_db_path, "metadata.json")
        if os.path.exists(metadata_path):
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            # 检查manual_chunks目录是否存在
            manual_chunks_dir = os.path.join(base_dir, "manual_chunks")
            if not os.path.exists(manual_chunks_dir):
                st.warning(f"文档目录不存在: {manual_chunks_dir}")
                os.makedirs(manual_chunks_dir, exist_ok=True)
            
            # 列出实际存在的文件
            existing_files = {}
            for root, _, files in os.walk(manual_chunks_dir):
                for file in files:
                    rel_path = os.path.join(os.path.relpath(root, base_dir), file)
                    existing_files[rel_path] = os.path.join(root, file)
                    # 也添加使用不同分隔符的路径
                    existing_files[rel_path.replace(os.sep, '\\')] = os.path.join(root, file)
                    existing_files[rel_path.replace(os.sep, '/')] = os.path.join(root, file)
            
            # 检查元数据中的文件路径
            path_fixed = False
            for item in metadata_list:
                if "path" in item:
                    path = item["path"]
                    
                    # 尝试修复路径
                    if path not in existing_files:
                        # 提取文件名
                        filename = os.path.basename(path.replace('\\', os.sep))
                        
                        # 查找匹配的文件
                        matching_files = [f for f in existing_files.keys() if f.endswith(filename)]
                        
                        if matching_files:
                            # 使用找到的第一个匹配文件
                            item["path"] = existing_files[matching_files[0]]
                            path_fixed = True
                            print(f"已修复路径: {path} -> {item['path']}")
                        else:
                            print(f"无法找到匹配文件: {path}")
                    else:
                        # 使用实际路径
                        item["path"] = existing_files[path]
            
            # 如果修复了路径，保存更新后的元数据
            if path_fixed:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                print("已保存修复后的元数据文件")
    
    except Exception as e:
        st.warning(f"检查文件路径时出错: {str(e)}")
    
    # 初始化RAG系统
    if 'rag_system' not in st.session_state or st.session_state.get('reload_rag', False):
        with st.spinner("正在加载知识库..."):
            try:
                st.session_state.rag_system = RAGSystem(vector_db_path)
                st.session_state.reload_rag = False
                st.success("知识库加载成功！")
            except Exception as e:
                st.error(f"加载RAG系统失败: {str(e)}")
                st.session_state.rag_system = None
                
                # 显示更详细的错误信息和解决方案
                st.error("""
                ### 可能的解决方法:
                1. 确保`vector_db_manual`目录中包含必要的文件
                2. 检查是否有权限访问该目录
                3. 如果是首次运行，请先添加文档到知识库
                4. 尝试重新启动应用
                """)
                
                # 提供一个重载按钮
                if st.button("尝试重新加载"):
                    st.session_state.reload_rag = True
                    st.rerun()
    
    # 创建文档处理器（保留基本功能）
    doc_processor = DocumentProcessor(base_dir)
    
    # 侧边栏
    with st.sidebar:
        st.title("🏛️ 政府采购与PPP智能咨询")
        
        # 使用更美观的图片显示
        st.markdown("""
        <div style="text-align:center; padding:10px;">
            <img src="https://img.huxiucdn.com/article/content/202306/05/222152647073.jpg" style="max-width:100%; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1);">
            <p style="font-size:0.8rem; color:#6B7280; margin-top:5px;">基于百度文心一言大模型提供服务</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 美化使用指南，修复图标对齐问题
        st.markdown("""
        <div style="background:linear-gradient(to right, #EFF6FF, #F9FAFB); padding:15px; border-radius:10px; margin:15px 0;">
            <h3 style="margin:0 0 10px 0; color:#1E40AF; font-size:1.2rem;">
                <span class="emoji-icon">💡</span> 使用指南
            </h3>
            <ol style="margin:0; padding-left:20px; color:#4B5563;">
                <li>在输入框中输入您关于政府采购或PPP项目的问题</li>
                <li>点击"查询解答"按钮获取专业回答</li>
                <li>查看参考来源了解信息出处</li>
                <li>历史记录中可查看之前的问答内容</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # 美化功能介绍，修复图标对齐问题
        st.markdown("""
        <div style="background:linear-gradient(to right, #EFF6FF, #F9FAFB); padding:15px; border-radius:10px;">
            <h3 style="margin:0 0 10px 0; color:#1E40AF; font-size:1.2rem;">
                <span class="emoji-icon">🔍</span> 核心功能
            </h3>
            <ul style="list-style-type:none; padding-left:5px; margin:0;">
                <li style="margin:10px 0;">
                    <span style="background:#3B82F6; color:white; padding:2px 8px; border-radius:10px; font-size:0.9rem;">检索增强</span>
                    <span style="margin-left:5px;">从专业文档中精准检索相关内容</span>
                </li>
                <li style="margin:10px 0;">
                    <span style="background:#3B82F6; color:white; padding:2px 8px; border-radius:10px; font-size:0.9rem;">专业解答</span>
                    <span style="margin-left:5px;">针对政府采购和PPP项目提供专业咨询</span>
                </li>
                <li style="margin:10px 0;">
                    <span style="background:#3B82F6; color:white; padding:2px 8px; border-radius:10px; font-size:0.9rem;">政策解读</span>
                    <span style="margin-left:5px;">解释最新政府政策法规和实施细则</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 仅保留智能问答功能
    st.markdown('<h2 class="main-header">政府采购和PPP项目智能问答系统</h2>', unsafe_allow_html=True)

    # 添加简短介绍，用更美观的方式展示
    st.markdown("""
    <div class="notice-box">
        <h3 style="margin-top:0; color:#1E40AF; font-size:1.3rem;">系统简介</h3>
        <p>本系统基于<strong>湖北省政府采购和PPP项目</strong>相关法规和政策文件，应用先进的检索增强生成技术(RAG)，可针对性回答相关专业问题。</p>
        <p>当知识库中没有查找到相关信息时，系统会切换至AI通用知识模式，同时明确标注来源，保证信息透明度。</p>
    </div>
    """, unsafe_allow_html=True)

    # 使用更加美观的提问表单
    with st.form("question_form"):
        # 添加一个图标和标题
        col_title1, col_title2 = st.columns([1, 20])
        with col_title1:
            st.markdown("💬")
        with col_title2:
            st.markdown("<h3 style='margin:0; font-size:1.3rem; color:#1E40AF;'>请输入您的专业问题</h3>", unsafe_allow_html=True)
        
        # 修复这里的空标签问题，添加标签并隐藏它
        query = st.text_area(
            label="问题输入", 
            value="",
            height=120,
            placeholder="例如：什么是政府采购？PPP项目的风险如何控制？政府采购供应商有哪些权利和义务？",
            label_visibility="collapsed"  # 隐藏标签但保留它以满足无障碍性要求
        )
        
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.markdown("<p class='info-text' style='font-size:0.9rem;'>提示: 问题越具体，回答越准确</p>", unsafe_allow_html=True)
        with col3:
            submit_button = st.form_submit_button("🔍 查询解答")

    # 处理问题提交
    if submit_button and query:
        if not st.session_state.rag_system:
            st.error("知识库加载失败，无法回答问题。请检查系统状态后重试。")
        else:
            with st.spinner("🧠 正在思考中..."):
                try:
                    # 调用RAG系统获取答案
                    result = st.session_state.rag_system.answer(query)
                    
                    # 显示结果
                    if "answer" in result:
                        # 显示问题，使用更美观的样式
                        st.markdown(f"<div class='sub-header'>您的问题</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='question-box'>{query}</div>", unsafe_allow_html=True)
                        
                        # 显示回答，添加标题
                        st.markdown(f"<div class='sub-header'>专业解答</div>", unsafe_allow_html=True)
                        st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                        
                        # 检查是否是来自知识库外的回答
                        if "【注意：知识库中没有找到与您问题直接相关的信息" in result["answer"]:
                            try:
                                # 分离免责声明和实际回答
                                disclaimer, answer_text = result["answer"].split("\n\n", 1)
                                st.markdown(f"<div class='disclaimer-box'>{disclaimer}</div>", unsafe_allow_html=True)
                                st.markdown(answer_text)
                            except ValueError:
                                # 如果无法拆分，直接显示完整回答
                                st.markdown(result["answer"])
                        else:
                            st.markdown(result["answer"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 显示参考来源
                        if "sources" in result and result["sources"]:
                            with st.expander("📚 参考来源详情", expanded=True):
                                for i, source in enumerate(result["sources"], 1):
                                    # 修复相关度计算，确保不为0
                                    similarity = source.get("similarity", source.get("score", 0))
                                    
                                    # 更全面的相似度处理逻辑
                                    if isinstance(similarity, (int, float)):
                                        if similarity < 0:
                                            # 如果是负值(可能是距离值)，将其转换为相关度分数
                                            # 使用更合理的转换方法
                                            relevance = max(10, min(95, 100 * (1 + similarity/10)))
                                        elif similarity == 0:
                                            # 确保至少有一个最小值，避免显示0%
                                            relevance = 15.0
                                        elif similarity < 1:
                                            # 如果是0-1范围的相似度值
                                            relevance = max(15, min(95, similarity * 100))
                                        else:
                                            # 如果是大于1的值(可能是原始分数)
                                            relevance = max(15, min(95, similarity * 10))
                                    else:
                                        relevance = 15.0  # 默认相关度
                                    
                                    relevance_color = "#10B981" if relevance > 70 else "#FBBF24" if relevance > 40 else "#EF4444"
                                    
                                    st.markdown(f"<div class='source-container'>", unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                                        <div class='source-title'>来源 {i}: {source.get('title', '未知来源')}</div>
                                        <div style='background-color:{relevance_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.8rem;'>
                                            相关度: {relevance:.1f}%
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # 获取内容
                                    content = source.get("content", "")
                                    
                                    # 添加内容总结和展示
                                    if content:
                                        # 展示原始内容（限制长度）
                                        summary_length = min(500, len(content))
                                        content_preview = content[:summary_length]
                                        if len(content) > summary_length:
                                            content_preview += "..."
                                        
                                        # 添加内容总结标签
                                        st.markdown("<div style='font-weight:500; margin-top:0.8rem; color:#4B5563;'>内容摘录：</div>", unsafe_allow_html=True)
                                        
                                        # 显示内容预览
                                        st.markdown(f"<div style='margin-top:0.4rem; background-color:#F9FAFB; padding:10px; border-radius:5px; font-size:0.95rem;'>{content_preview}</div>", unsafe_allow_html=True)
                                        
                                        # 为长文本添加"查看完整内容"按钮
                                        if len(content) > summary_length:
                                            with st.expander("查看完整内容"):
                                                st.markdown(f"<div style='padding:10px;'>{content}</div>", unsafe_allow_html=True)
                                        
                                        # 添加内容总结（使用短句描述文档内容要点）
                                        if len(content) > 200:  # 只为较长文档生成总结
                                            summary = f"该文档主要涉及{source.get('title', '相关政策')}的内容，包含约{len(content)}个字符的专业信息。"
                                            st.markdown(f"<div style='margin-top:0.8rem; font-style:italic; color:#4B5563; font-size:0.9rem;'>📝 内容简介：{summary}</div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("<div style='margin-top:0.8rem; color:#EF4444;'>无可用内容</div>", unsafe_allow_html=True)
                                    
                                    if "path" in source:
                                        st.caption(f"📄 文件路径: {source['path']}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("⚠️ 未找到相关参考来源")
                        
                        # 显示性能指标，使用更美观的卡片
                        if "metrics" in result:
                            st.markdown("<div class='sub-header' style='font-size:1.2rem;'>系统性能指标</div>", unsafe_allow_html=True)
                            metrics = result["metrics"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>⏱️ 检索耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('retrieval_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with col2:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>⚙️ 生成耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('generation_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with col3:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>🕒 总耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('total_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                
                    # 保存到历史记录
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        "query": query,
                        "result": result,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"处理问题时出错: {str(e)}")
                    import traceback
                    st.exception(e)
                    st.error(f"详细错误信息: {traceback.format_exc()}")

    # 显示历史问答记录
    if st.session_state.get('history'):
        with st.expander("📜 历史问答记录", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history)):
                st.markdown(f"<div class='history-item'>", unsafe_allow_html=True)
                
                # 添加时间戳和序号
                timestamp = item.get('timestamp', '')
                
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                    <div class='history-question'>问题 {len(st.session_state.history) - i}：{item['query']}</div>
                    <div style='font-size:0.8rem; color:#6B7280;'>{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 只显示回答的前150个字符作为摘要
                answer = item['result'].get('answer', '无回答')
                if len(answer) > 150:
                    answer_summary = answer[:150] + "..."
                else:
                    answer_summary = answer
                
                st.markdown(f"<div class='history-answer'>回答：{answer_summary}</div>", unsafe_allow_html=True)
                
                # 添加"查看完整回答"按钮
                if st.button(f"查看完整回答 #{len(st.session_state.history) - i}", key=f"view_{i}"):
                    st.session_state.query = item['query']
                    st.rerun()
                    
                st.markdown("</div>", unsafe_allow_html=True)
    
    # 页脚
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-flex">
                <div class="footer-item" style="font-weight:500;">政府采购和PPP项目智能问答系统</div>
                <div class="footer-item divider">|</div>
                <div class="footer-item">© 2025</div>
            </div>
            <div style="font-size:0.8rem; margin-bottom:8px;">基于检索增强生成(RAG)技术与百度文心一言大模型构建</div>
            <div style="margin-top:10px; font-size:0.9rem; font-weight:500; color:#1E40AF;">采招云（湖北）信息科技有限公司</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 确保只有在作为主模块运行时才调用main函数
if __name__ == "__main__":
    main()