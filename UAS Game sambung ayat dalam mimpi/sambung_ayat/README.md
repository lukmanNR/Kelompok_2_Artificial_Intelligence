# Sambung Ayat Al-Qur'an — Mimpi Pocong Edition

Game edukasi Islami: sambung ayat Juz 30 + cerita visual (cewek ketiduran & pocong).

Dibangun sesuai arsitektur tugas:

```
User → FE → BE (FastAPI)
              ↓
         LangGraph pipeline
           retrieve → select → format
              ↓
         AI Source (Hybrid)
           ├── Ollama (local)  → primary
           └── Google Gemini   → fallback
              ↓
         Vector DB (ChromaDB)
              ↓
         data/surahs/*.txt  (sumber Al-Qur'an)
```

---

## 1. Fitur Gameplay

1. Pilih Surah Juz 30
2. Muncul Ayat N → pilih lanjutan yang benar (A/B)
3. Timer **9 detik** per soal
4. **5 benar ** → Good Ending
5. Salah / timeout ke-1 → pocong masuk kamar
6. Salah / timeout ke-2 → layar hitam 2.7 detik → terbangun di kelas (4 detik) → ending buruk
7. Ending cerita 5 detik → layar **Tamat** + tombol Main Lagi

**Layout UI:** kiri 75% gambar | kanan 25% panel soal

---

## 2. Arsitektur (sesuai diagram)

| Komponen | Implementasi |
|----------|----------------|
| **FE** | `frontend/index.html` (HTML + Tailwind + JS) |
| **BE** | `backend/main.py` (FastAPI + Uvicorn) |
| **LangGraph** | Node: `retrieve` → `select` → `format` |
| **RAG** | txt → chunk per ayat → embed → Chroma → retrieve |
| **Vector DB** | ChromaDB (`data/chroma_db/`) |
| **Ollama** | Primary AI Source (`qwen2.5:3b`) |
| **Gemini** | Fallback AI Source (`gemini-1.5-flash`) |
| **Anti-halu** | SLM hanya return `{"selected_distractor_id": N}` — teks Arab 100% dari data |

### Alur RAG (ideal yang dijalankan)

1. **File sumber** → `data/surahs/*.txt` (1 file per surah)
2. **Chunking** → 1 ayat = 1 chunk (`NO|ARAB|TERJEMAHAN`)
3. **Embedding** → Ollama (`nomic-embed-text`)
4. **Vector DB** → ChromaDB
5. **Retrieve** → ambil kandidat distractor (filter surah + aturan)
6. **SLM/LLM Select** → Ollama / Gemini pilih ID saja (bukan generate Arab)

### Filter soal (ketat)

- Tidak boleh Bismillah sebagai soal
- Tidak boleh ayat terakhir surah sebagai soal
- Tidak boleh ayat yang teks Arabnya identik berulang

---

## 3. Struktur Folder

```
sambung_ayat/
├── backend/
│   └── main.py              # FastAPI + LangGraph + Hybrid AI + RAG
├── frontend/
│   └── index.html           # UI game
├── data/
│   ├── surahs/              # File sumber Al-Qur'an (.txt per surah)
│   │   ├── 081_At-Takwir.txt
│   │   ├── ...
│   │   └── 114_An-Nas.txt
│   ├── chroma_db/           # Vector DB (otomatis terisi saat start)
│   └── quran_sample.json    # (legacy, tidak dipakai pipeline utama)
├── assets/
│   ├── bedroom_sleep.png
│   ├── bedroom_pocong_in.png
│   ├── classroom_wake.png
│   └── dream_photo.png
├── requirements.txt
└── README.md
```

---

## 4. Instalasi

```powershell
cd sambung_ayat

pip install -r requirements.txt

# Model Ollama (wajib)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Gemini (opsional, untuk hybrid penuh)

1. Buat API key: https://aistudio.google.com/apikey
2. Set di PowerShell:

```powershell
$env:GEMINI_API_KEY="AIzaSy...."
```

---

## 5. Menjalankan

**Terminal 1 — Ollama**
```powershell
ollama serve
```

**Terminal 2 — Backend**
```powershell
cd sambung_ayat
$env:GEMINI_API_KEY="AIzaSy...."   # opsional
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka browser: **http://127.0.0.1:8000**

Hard refresh jika perlu: `Ctrl + Shift + R`

### Cek status sistem

http://127.0.0.1:8000/api/health

Contoh output sehat:
```json
{
  "status": "ok",
  "ollama_ok": true,
  "langgraph_installed": true,
  "gemini_configured": true,
  "chroma_ok": true,
  "pipeline": "LangGraph: retrieve → select(Ollama|Gemini|rule) → format"
}
```

---

## 6. Asset Gambar

Letakkan di `assets/` dengan nama **persis**:

| File | Kapan muncul |
|------|----------------|
| `bedroom_sleep.png` | Awal game + good ending |
| `bedroom_pocong_in.png` | Setelah salah ke-1 |
| `classroom_wake.png` | Setelah salah ke-2 |
| `dream_photo.png` | Ending buruk (pocong selimutan) |

---

## 7. Menambah Surah

Buat file di `data/surahs/` format:

```
# Surah 112 | Al-Ikhlas | الإخلاص
# total_ayat=4

1|قُلْ هُوَ اللَّهُ أَحَدٌ|Katakanlah (Muhammad), "Dialah Allah, Yang Maha Esa."
2|اللَّهُ الصَّمَدُ|Allah tempat meminta segala sesuatu.
3|لَمْ يَلِدْ وَلَمْ يُولَدْ|Dia tidak beranak dan tidak diperanakkan.
4|وَلَمْ يَكُنْ لَهُ كُفُوًا أَحَدٌ|Dan tidak ada sesuatu yang setara dengan Dia.
```

Restart server → surah otomatis masuk.

---

## 8. Penjelasan ke Dosen (siap pakai)

> Aplikasi ini mengimplementasikan arsitektur RAG dengan LangGraph.  
> Data Al-Qur’an disimpan sebagai file teks per surah, di-chunk per ayat, di-embed menggunakan Ollama, lalu disimpan di ChromaDB.  
> Pipeline LangGraph: **retrieve → select → format**.  
> AI Source bersifat hybrid: **Ollama lokal (primary)** dan **Google Gemini (fallback)**.  
> SLM hanya diperbolehkan mengembalikan ID distractor dalam JSON; seluruh teks Arab diambil dari knowledge base (anti-halusinasi).  
> Frontend menampilkan game visual berbasis state (kamar → pocong masuk → kelas → ending).

---

## 9. Catatan jujur (hindari overclaim)

- LangGraph dipakai jika library terinstall; jika tidak, node yang sama dijalankan sequential (hasil identik).
- Gemini hanya aktif jika `GEMINI_API_KEY` di-set.
- Indexing Chroma + embed berjalan di background saat startup (bisa 1–5 menit pertama kali).
- Jeda singkat antar soal = waktu inferensi Ollama (normal untuk model 3B di CPU).
