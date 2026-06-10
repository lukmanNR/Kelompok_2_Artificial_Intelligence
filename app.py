import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Konfigurasi Tampilan Halaman Streamlit
st.set_page_config(page_title="Tanya Rezeki AI", page_icon="🕌")
st.title("🕌 Chatbot Tanya Jawab Rezeki")
st.caption("Berdasarkan pemahaman Islam terdahulu (Sumber: Rumaysho, Tafsirweb, dsb)")

# 2. Inisialisasi Database dan Model AI
@st.cache_resource
def load_system():
    # Load database yang sudah dibuat oleh ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Panggil model Qwen 2.5 3B via Ollama
    llm = Ollama(model="qwen2.5:3b")
    
    # 3. Buat Prompt Khusus
    system_prompt = (
        "Kamu adalah seorang Ustadz dan asisten yang ahli dalam memberikan nasihat agama terkait rezeki berdasarkan pemahaman Salafus Shalih.\n"
        "Gunakan konteks di bawah ini untuk menjawab pertanyaan.\n"
        "JIKA JAWABANNYA TIDAK ADA DI DALAM KONTEKS, katakan 'Wallahu a'lam, saya tidak menemukan pembahasan tersebut di dalam referensi yang saya miliki.'\n"
        "JANGAN mengarang dalil atau ayat sendiri. Jawab dengan bahasa yang santun, jelas, dan menguatkan tauhid.\n\n"
        "Konteks:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Fungsi pembantu untuk merapikan teks dokumen
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 4. MEMBANGUN RAG DENGAN LCEL (Tanpa menggunakan langchain.chains!)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

rag_chain = load_system()

# 5. Fitur Chat Streamlit (Menyimpan riwayat obrolan)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom Input Chat
if user_input := st.chat_input("Tanyakan soal rezeki (misal: Apakah rezeki bisa tertukar?)"):
    # Tampilkan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proses AI berpikir
    with st.chat_message("assistant"):
        with st.spinner("Mencari dalil dan referensi..."):
            try:
                # Jalankan RAG modern
                jawaban = rag_chain.invoke(user_input)
                st.markdown(jawaban)
                st.session_state.messages.append({"role": "assistant", "content": jawaban})
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}. Pastikan Ollama sudah berjalan.")