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

# 设置页面主题
def set_page_theme():
    # 自定义CSS样式
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            margin-bottom: 1rem;
            text-align: center;
            padding: 1.5rem 0;
            border-bottom: 2px solid #E5E7EB;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #1E3A8A;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #E5E7EB;
        }
        .answer-container {
            background-color: #F3F4F6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 5px solid #3B82F6;
        }
        .info-text {
            color: #4B5563;
            font-size: 1rem;
            line-height: 1.5;
        }
        .highlight-text {
            color: #1E3A8A;
            font-weight: bold;
        }
        .source-title {
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #1E3A8A;
        }
        .source-container {
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }
        .metrics-card {
            background-color: #F9FAFB;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metrics-label {
            font-size: 0.8rem;
            color: #4B5563;
            margin-bottom: 0.3rem;
        }
        .metrics-value {
            font-size: 1.2rem;
            color: #1E3A8A;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #6B7280;
            font-size: 0.8rem;
            margin-top: 2rem;
            border-top: 1px solid #E5E7EB;
        }
        /* 自定义进度条样式 */
        .stProgress > div > div > div > div {
            background-color: #3B82F6;
        }
        /* 自定义按钮样式 */
        .stButton button {
            background-color: #1E3A8A;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #2563EB;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        /* 自定义侧边栏样式 */
        .css-1d391kg {
            background-color: #F9FAFB;
        }
        /* 上传区域样式 */
        .upload-section {
            background-color: #F3F4F6;
            padding: 1.2rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border: 1px dashed #3B82F6;
        }
        .success-message {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 5px solid #10B981;
        }
        .parse-section {
            background-color: #F5F3FF;
            padding: 1.2rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border: 1px dashed #8B5CF6;
        }
        .parse-result {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border-left: 4px solid #8B5CF6;
            font-family: 'Georgia', serif;
            white-space: pre-wrap;
        }
        .tab-content {
            padding: 1rem 0;
        }
        .parse-header {
            color: #6D28D9;
            font-weight: bold;
            margin-bottom: 0.5rem;
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
        st.title("政府采购和PPP项目智能问答")
        st.image("https://img.huxiucdn.com/article/content/202306/05/222152647073.jpg", use_column_width=True)
        
        st.markdown("### 使用指南")
        st.markdown("""
        <div class="info-text">
        1. 在文本框中输入您的问题
        2. 点击"提交问题"按钮
        3. 系统将检索相关文档并生成回答
        4. 可查看参考来源和系统性能指标
        </div>
        """, unsafe_allow_html=True)
        
        # 更新功能描述
        st.markdown("### 功能介绍")
        st.markdown("""        
        <div class="info-text">
        <span class="highlight-text">智能问答功能</span>：针对政府采购和PPP项目的专业问答系统
        </div>
        """, unsafe_allow_html=True)
        
        # 保留系统状态检查等其他功能
        # ...
    
    # 仅保留智能问答功能
    st.markdown('<h2 class="main-header">政府采购和PPP项目智能问答</h2>', unsafe_allow_html=True)

    # 智能问答功能实现
    # 添加输入框和提交按钮
    with st.form("question_form"):
        query = st.text_area("请输入您的问题:", height=100, 
                              placeholder="例如：什么是政府采购？或者 PPP项目的风险如何控制？")
        
        submit_button = st.form_submit_button("提交问题")

    # 处理问题提交
    if submit_button and query:
        if not st.session_state.rag_system:
            st.error("知识库加载失败，无法回答问题。请检查系统状态后重试。")
        else:
            with st.spinner("正在思考，请稍候..."):
                try:
                    # 调用RAG系统获取答案
                    result = st.session_state.rag_system.answer(query)
                    
                    # 显示结果
                    if "answer" in result:
                        st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                        st.markdown(result["answer"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 显示参考来源
                        if "sources" in result and result["sources"]:
                            with st.expander("参考来源", expanded=True):
                                for i, source in enumerate(result["sources"], 1):
                                    st.markdown(f"<div class='source-container'>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='source-title'>来源 {i}: {source.get('title', '未知来源')}</div>", unsafe_allow_html=True)
                                    st.markdown(source.get("content", ""))
                                    if "path" in source:
                                        st.caption(f"文件路径: {source['path']}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("未找到相关参考来源")
                        
                        # 显示性能指标
                        if "metrics" in result:
                            metrics = result["metrics"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>检索耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('retrieval_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with col2:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>生成耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('generation_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with col3:
                                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='metrics-label'>总耗时</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metrics-value'>{metrics.get('total_time', 0):.3f}秒</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 保存到历史记录
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        "query": query,
                        "result": result
                    })
                    
                except Exception as e:
                    st.error(f"处理问题时出错: {str(e)}")
                    st.exception(e)

    # 显示历史问答记录
    if st.session_state.get('history'):
        with st.expander("历史问答记录", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**问题 {len(st.session_state.history) - i}**: {item['query']}")
                st.markdown(f"**回答**: {item['result'].get('answer', '无回答')}")
                st.markdown("---")
    
    # 页脚
    st.markdown("""
    <div class="footer">
        政府采购和PPP项目智能问答系统 © 2025 | 基于RAG技术构建
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()