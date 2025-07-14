# rag-streamlit-app/rag-streamlit-app/README.md

# RAG Streamlit Application

这是一个基于RAG（Retrieval-Augmented Generation）系统的Streamlit应用，旨在提供政府采购和PPP项目相关问题的智能问答服务。

## 项目结构

```
rag-streamlit-app
├── app.py                # Streamlit应用的主文件
├── rag_system.py         # RAG系统实现文件
├── requirements.txt      # 项目所需的Python库和依赖项
├── vector_db_manual      # 向量数据库相关文件夹
│   ├── index.faiss       # FAISS索引文件
│   └── metadata.json     # 文本元数据文件
└── README.md             # 项目文档
```

## 安装依赖

在运行应用之前，请确保安装了所需的Python库。您可以使用以下命令安装依赖项：

```
pip install -r requirements.txt
```

## 运行应用

要启动Streamlit应用，请在终端中运行以下命令：

```
streamlit run app.py
```

然后，打开浏览器并访问 `http://localhost:8501` 以使用应用。

## 使用说明

1. 在输入框中输入您的问题。
2. 点击“提交”按钮。
3. 应用将检索相关文档并生成回答。
4. 查看生成的回答和相关来源。

## 贡献

欢迎任何形式的贡献！如果您有建议或发现问题，请提交问题或拉取请求。

## 许可证

此项目采用MIT许可证。有关详细信息，请参阅LICENSE文件。