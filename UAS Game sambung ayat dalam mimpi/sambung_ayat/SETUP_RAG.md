# Setup RAG Asli — Sambung Ayat

## Arsitektur yang dijalankan

```
data/surahs/*.txt          ← File sumber (1 file per surah)
        ↓
   Chunking (1 ayat = 1 chunk)
        ↓
   Embedding (Ollama: nomic-embed-text)
        ↓
   ChromaDB (data/chroma_db/)
        ↓
   Similarity Retrieve
        ↓
   SLM Selector (Ollama: qwen2.5:3b) → hanya pilih ID
        ↓
   Response ke Frontend
```

## Install (Windows PowerShell)

```powershell
cd "D:\Heaven Path\Tazkia\tugas tazkia\pak hendri\4\AI\sambung_ayat"

pip install fastapi uvicorn httpx pydantic chromadb

# Model embedding (WAJIB untuk RAG asli)
ollama pull nomic-embed-text

# Model SLM (sudah ada)
ollama list
# pastikan qwen2.5:3b ada
```

## Jalankan

```powershell
# Terminal 1: pastikan Ollama hidup
ollama serve

# Terminal 2: backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka: http://127.0.0.1:8000

Cek status RAG: http://127.0.0.1:8000/api/health

Yang harus `true`:
- ollama_ok
- chroma_ok (setelah index selesai)
- embed_model_ready (setelah `ollama pull nomic-embed-text`)

## Catatan pertama kali

Saat server pertama kali start, backend akan:
1. Baca semua file di `data/surahs/*.txt`
2. Embed setiap ayat lewat Ollama
3. Simpan ke ChromaDB

Ini bisa memakan 1–5 menit tergantung jumlah ayat.
