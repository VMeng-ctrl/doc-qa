# app.py —— 项目2：文档问答系统（网页版）
import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

base = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Lumina 手册问答", page_icon="📄")

# ===== 备菜：索引只建一次，之后所有提问都复用 =====
@st.cache_resource
def build_index():
    reader = PdfReader(os.path.join(base, "lumina_manual.pdf"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks, metadatas = [], []
    for i, page in enumerate(reader.pages):
        for p in splitter.split_text(page.extract_text()):
            chunks.append(p)
            metadatas.append({"page": i + 1})
    # 双保险：本地有缓存就离线用，云上没缓存就联网下载
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return Chroma.from_texts(
        texts=chunks, embedding=embeddings,
        metadatas=metadatas, collection_name="manual",
    )

store = build_index()

# ===== API key：本地读 .env，云上读 Streamlit Secrets（部署时的伏笔）=====
def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        load_dotenv(os.path.expanduser("~/Projects/AI-Study/代码练习/.env"))
        return os.getenv("DEEPSEEK_API_KEY")

# ===== 页面零件（从上到下就是你点的兵）=====
st.title("📄 Lumina Notes 手册问答")
st.caption("基于 RAG：答案只来自手册，并标注来源页码")

question = st.text_input("请输入问题（例如：What is the refund policy?）")

if st.button("提问"):
    if not question.strip():
        st.warning("问题不能是空的哦")
    else:
        with st.spinner("检索手册并思考中..."):
            # 搜索（带页码身份证）
            hits = store.similarity_search(question, k=2)
            context = "\n\n".join(
                [f"(第{h.metadata['page']}页) {h.page_content}" for h in hits]
            )
            prompt = f"""只根据下面的资料回答问题，用中文回答。
如果资料里没有答案，就老实说"资料里没找到"，并在最后一行写"来源：无"。
如果有答案，最后一行单独写"来源：第X页"。

资料：
{context}

问题：{question}"""
            client = OpenAI(api_key=get_api_key(), base_url="https://api.deepseek.com/v1")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content

        # 把回答正文和"来源行"拆开显示
        lines = [l for l in answer.strip().split("\n") if l.strip()]
        来源行 = [l for l in lines if l.strip().strip("*").startswith("来源")]
        正文 = [l for l in lines if l not in 来源行]

        st.write("\n".join(正文))          # 公示栏：回答
        if 来源行:
            st.info(来源行[-1])            # 高亮：来源（蓝色提示框）
        elif "没找到" in answer:
            st.info("来源：无")

        # 抽屉：检索依据
        with st.expander("🔍 查看检索依据"):
            for i, h in enumerate(hits):
                st.markdown(f"**第{i+1}名 · 来自第{h.metadata['page']}页**")
                st.write(h.page_content)