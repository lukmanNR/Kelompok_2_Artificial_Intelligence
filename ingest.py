from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 1. Baca semua PDF di folder dataset
print("Memulai proses membaca ratusan PDF... (Ini mungkin memakan waktu beberapa menit)")
loader = DirectoryLoader('./dataset', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Alhamdulillah, {len(documents)} halaman berhasil dibaca.")

# 2. Potong teks menjadi bagian kecil agar AI mudah mencernanya
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Teks berhasil dipotong menjadi {len(texts)} bagian.")

# 3. Ubah teks menjadi vektor (angka) menggunakan model HuggingFace gratis
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Simpan ke dalam folder 'chroma_db' sebagai database permanen
print("Menyimpan ke Vector Database...")
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

print("Proses Selesai! Database berhasil dibuat di folder 'chroma_db'.")