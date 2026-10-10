# 📄 文档问答助手（doc-qa）

内置一份公司内部标准手册，向它提问，AI 读完文档直接给出答案，并标注出处页数。

## ✨ 功能

- 在输入框输入关于这份手册的任何问题
- AI 阅读文档内容，直接给出答案（不是丢一堆链接让你自己翻）
- 答案**标明来源页数**，答错可以立刻对质
- 当前文档为内置手册（`lumina_manual.pdf`），更换文档需修改代码

## 🌐 在线试用

**https://doc-app-yxse4dpa6h5rrekqzsncgx.streamlit.app/**

## 🚀 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 工作原理（RAG）

```
PDF → 切成小块 → 向量化（SentenceTransformer） → 存入 ChromaDB
                                                   ↓
用户提问 → 问题向量化 → 检索最相关的几块 → 连同问题一起交给 DeepSeek → 带页码出处的答案
```

关键点：模型没见过这份文档，是**检索到的原文**喂给它，它照着资料回答——所以必须标出处，避免"用记忆冒充证据"。

## 🛠 技术栈

Python 3.14 · LangChain（切块与向量库封装）· DeepSeek API · ChromaDB（向量库）· HuggingFaceEmbeddings / sentence-transformers（文本转向量）· Streamlit（界面）
