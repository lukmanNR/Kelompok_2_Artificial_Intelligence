"""
Sambung Ayat - Mimpi Pocong
Arsitektur sesuai diagram:
  User → FE → BE → AI Source (Hybrid: Ollama primary + Gemini fallback)
                 → LangGraph pipeline (retrieve → select → format)
                 → Vector DB (Chroma)
                 → data/surahs/*.txt

Anti-halu: SLM HANYA return ID distractor, tidak generate teks Arab.
"""

import json
import random
import re
import hashlib
import os
import threading
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from pathlib import Path
from collections import Counter

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------- PATHS & CONFIG --------------------
BASE = Path(__file__).parent.parent
SURAHS_DIR = BASE / "data" / "surahs"
CHROMA_DIR = BASE / "data" / "chroma_db"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
SLM_MODEL = "qwen2.5:3b"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
GEMINI_MODEL = "gemini-1.5-flash"

app = FastAPI(title="Sambung Ayat RAG + LangGraph API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# -------------------- MODELS --------------------
class SurahInfo(BaseModel):
    surah_id: int
    surah_name: str
    surah_name_ar: str
    total_ayat: int

class AyatOut(BaseModel):
    no: int
    arab: str
    id: str

class QuestionResponse(BaseModel):
    question_id: str
    surah_id: int
    surah_name: str
    ayat_soal: AyatOut
    pilihan: List[AyatOut]
    correct_index: int
    ai_source: str

class AnswerRequest(BaseModel):
    question_id: str
    selected_index: int
    surah_id: int

class AnswerResponse(BaseModel):
    correct: bool
    explanation: str
    next_ready: bool

# -------------------- LOAD TXT (sumber RAG) --------------------
def load_surahs_from_txt() -> List[dict]:
    surahs = []
    if not SURAHS_DIR.exists():
        print("[Load] folder data/surahs tidak ada")
        return surahs
    for path in sorted(SURAHS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        meta = {"surah_id": 0, "surah_name": path.stem, "surah_name_ar": "", "ayat": []}
        for ln in lines:
            if ln.startswith("# Surah"):
                parts = [p.strip() for p in ln.replace("# Surah", "").split("|")]
                try:
                    meta["surah_id"] = int(parts[0])
                except Exception:
                    pass
                if len(parts) > 1:
                    meta["surah_name"] = parts[1]
                if len(parts) > 2:
                    meta["surah_name_ar"] = parts[2]
            elif ln.startswith("#"):
                continue
            else:
                bits = ln.split("|", 2)
                if len(bits) >= 3:
                    try:
                        no = int(bits[0])
                    except Exception:
                        continue
                    meta["ayat"].append({"no": no, "arab": bits[1].strip(), "id": bits[2].strip()})
        if meta["surah_id"] and meta["ayat"]:
            surahs.append(meta)
    surahs.sort(key=lambda s: s["surah_id"])
    print(f"[Load] {len(surahs)} surah, {sum(len(s['ayat']) for s in surahs)} ayat")
    return surahs

QURAN_DATA = load_surahs_from_txt()
SURAH_MAP = {s["surah_id"]: s for s in QURAN_DATA}

# -------------------- EMBEDDING --------------------
def ollama_embed(text: str) -> Optional[List[float]]:
    try:
        with httpx.Client(timeout=20.0) as client:
            for model in (EMBED_MODEL, SLM_MODEL):
                r = client.post(f"{OLLAMA_BASE}/api/embeddings",
                                json={"model": model, "prompt": text})
                if r.status_code == 200:
                    emb = r.json().get("embedding")
                    if emb:
                        return emb
    except Exception as e:
        print(f"[Embed] {e}")
    return None

# -------------------- VECTOR DB --------------------
COLLECTION = None
CHROMA_OK = False
MEMORY_STORE: List[dict] = []

def init_memory():
    global MEMORY_STORE
    MEMORY_STORE = []
    for surah in QURAN_DATA:
        for ayat in surah["ayat"]:
            MEMORY_STORE.append({
                "id": f"{surah['surah_id']}_{ayat['no']}",
                "surah_id": surah["surah_id"],
                "surah_name": surah["surah_name"],
                "no": ayat["no"],
                "arab": ayat["arab"],
                "translation": ayat["id"],
                "document": f"{ayat['arab']} | {ayat['id']}",
            })

def init_chroma():
    global COLLECTION, CHROMA_OK
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            client.delete_collection("quran_ayat")
        except Exception:
            pass
        COLLECTION = client.create_collection("quran_ayat", metadata={"hnsw:space": "cosine"})
        ids, docs, metas, embeds = [], [], [], []
        for m in MEMORY_STORE:
            emb = ollama_embed(m["document"])
            if emb is None:
                h = hashlib.sha256(m["document"].encode()).digest()
                emb = [((b / 255.0) * 2 - 1) for b in (h * 16)][:384]
            ids.append(m["id"])
            docs.append(m["document"])
            metas.append({
                "surah_id": m["surah_id"], "surah_name": m["surah_name"],
                "no": m["no"], "arab": m["arab"], "translation": m["translation"],
            })
            embeds.append(emb)
        COLLECTION.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        CHROMA_OK = True
        print(f"[Chroma] {len(ids)} ayat indexed")
    except Exception as e:
        print(f"[Chroma] {e}")
        CHROMA_OK = False

init_memory()
threading.Thread(target=init_chroma, daemon=True).start()

# -------------------- FILTERS --------------------
def is_bismillah(text: str) -> bool:
    t = text.strip()
    return "بِسْمِ اللَّهِ" in t or "بسم الله" in t or t.startswith("بِسْمِ")

def get_valid_soal(surah: dict) -> List[dict]:
    ayat_list = surah["ayat"]
    if len(ayat_list) < 2:
        return []
    cnt = Counter(a["arab"].strip() for a in ayat_list)
    out = []
    for a in ayat_list[:-1]:
        if is_bismillah(a["arab"]):
            continue
        if cnt[a["arab"].strip()] > 1:
            continue
        out.append(a)
    return out

# -------------------- RETRIEVE --------------------
def retrieve_candidates(surah_id: int, exclude_nos: List[int], k: int = 6) -> List[dict]:
    pool = [
        m for m in MEMORY_STORE
        if m["surah_id"] == surah_id and m["no"] not in exclude_nos and not is_bismillah(m["arab"])
    ]
    random.shuffle(pool)
    if pool:
        return pool[:k]
    pool = [m for m in MEMORY_STORE if m["surah_id"] != surah_id and not is_bismillah(m["arab"])]
    random.shuffle(pool)
    return pool[:k]

# -------------------- AI SOURCE: Ollama + Gemini --------------------
def _parse_id(raw: str, n: int) -> Optional[int]:
    m = re.search(r'"selected_distractor_id"\s*:\s*(\d+)', raw)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n:
            return idx
    return None

def ollama_select(correct_arab: str, candidates: List[dict]) -> Optional[int]:
    cands = candidates[:4]
    options = "\n".join(f"{i}: {c['arab']}" for i, c in enumerate(cands))
    prompt = (
        f"Pilih index distractor paling mirip akhiran (fawasil) dengan target.\n"
        f"Target: {correct_arab}\nKandidat:\n{options}\n"
        f'Jawab HANYA: {{"selected_distractor_id": N}}'
    )
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post(f"{OLLAMA_BASE}/api/generate", json={
                "model": SLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 40, "num_ctx": 512},
            })
            if r.status_code == 200:
                return _parse_id(r.json().get("response", ""), len(cands))
    except Exception as e:
        print(f"[Ollama] {e}")
    return None

def gemini_select(correct_arab: str, candidates: List[dict]) -> Optional[int]:
    """Fallback AI Source: Google Gemini."""
    if not GEMINI_API_KEY:
        return None
    cands = candidates[:4]
    options = "\n".join(f"{i}: {c['arab']}" for i, c in enumerate(cands))
    prompt = (
        f"Pilih index distractor paling mirip akhiran (fawasil) dengan target.\n"
        f"Target: {correct_arab}\nKandidat:\n{options}\n"
        f'Jawab HANYA JSON: {{"selected_distractor_id": N}}'
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 40},
            })
            if r.status_code != 200:
                print(f"[Gemini] HTTP {r.status_code}: {r.text[:120]}")
                return None
            data = r.json()
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_id(raw, len(cands))
    except Exception as e:
        print(f"[Gemini] {e}")
    return None

def rule_fallback(correct_arab: str, candidates: List[dict]) -> int:
    def tail(t, n=4):
        c = re.sub(r"[^\u0600-\u06FF]", "", t)
        return c[-n:] if len(c) >= n else c
    target = tail(correct_arab)
    scored = []
    for i, c in enumerate(candidates):
        end = tail(c["arab"])
        sc = 10 if end == target else (6 if len(end) >= 2 and end[-2:] == target[-2:] else 0)
        scored.append((sc, i))
    scored.sort(key=lambda x: (-x[0], random.random()))
    return scored[0][1]

def hybrid_ai_select(correct_arab: str, candidates: List[dict]) -> Tuple[dict, str]:
    """
    AI Source Hybrid (sesuai diagram):
      1. Ollama local (primary)
      2. Google Gemini (fallback)
      3. Rule-based (last resort)
    """
    idx = ollama_select(correct_arab, candidates)
    if idx is not None:
        return candidates[idx], "ollama"
    idx = gemini_select(correct_arab, candidates)
    if idx is not None:
        return candidates[idx], "gemini"
    idx = rule_fallback(correct_arab, candidates)
    return candidates[idx], "fallback"

# -------------------- LANGGRAPH PIPELINE --------------------
# State untuk LangGraph
class GraphState(TypedDict, total=False):
    surah_id: int
    soal: dict
    correct: dict
    candidates: List[dict]
    distractor: dict
    ai_source: str
    choices: List[dict]
    correct_index: int
    question_id: str
    error: str

def node_retrieve(state: GraphState) -> GraphState:
    """LangGraph node: Retrieve kandidat dari Vector/Memory store."""
    sid = state["surah_id"]
    soal = state["soal"]
    correct = state["correct"]
    cands = retrieve_candidates(sid, [soal["no"], correct["no"]], k=6)
    return {**state, "candidates": cands}

def node_select(state: GraphState) -> GraphState:
    """LangGraph node: AI Source hybrid pilih distractor."""
    distractor, source = hybrid_ai_select(state["correct"]["arab"], state["candidates"])
    return {**state, "distractor": distractor, "ai_source": source}

def node_format(state: GraphState) -> GraphState:
    """LangGraph node: format pilihan A/B (teks Arab dari data asli saja)."""
    correct = state["correct"]
    distractor = state["distractor"]
    choices = [
        {"no": correct["no"], "arab": correct["arab"], "id": correct["id"]},
        {
            "no": distractor["no"],
            "arab": distractor["arab"],
            "id": distractor.get("translation", distractor.get("id", "")),
        },
    ]
    random.shuffle(choices)
    correct_index = 0 if choices[0]["arab"] == correct["arab"] else 1
    qid = f"{state['surah_id']}_{state['soal']['no']}_{random.randint(1000,9999)}"
    return {
        **state,
        "choices": choices,
        "correct_index": correct_index,
        "question_id": qid,
    }

def run_langgraph_pipeline(surah_id: int) -> Optional[dict]:
    """
    Menjalankan pipeline ala LangGraph:
      retrieve → select (AI hybrid) → format
    Jika library langgraph terinstall, pakai StateGraph.
    Jika tidak, jalankan node secara sequential (hasil sama).
    """
    surah = SURAH_MAP.get(surah_id)
    if not surah:
        return None
    valid = get_valid_soal(surah)
    if not valid:
        return None
    soal = random.choice(valid)
    correct = next(a for a in surah["ayat"] if a["no"] == soal["no"] + 1)

    init_state: GraphState = {
        "surah_id": surah_id,
        "soal": soal,
        "correct": correct,
    }

    # Coba LangGraph asli
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(GraphState)
        g.add_node("retrieve", node_retrieve)
        g.add_node("select", node_select)
        g.add_node("format", node_format)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "select")
        g.add_edge("select", "format")
        g.add_edge("format", END)
        app_graph = g.compile()
        final = app_graph.invoke(init_state)
        print("[LangGraph] pipeline OK")
    except Exception as e:
        # Sequential fallback (node yang sama)
        print(f"[LangGraph] lib tidak ada / error ({e}) → sequential nodes")
        s = node_retrieve(init_state)
        s = node_select(s)
        final = node_format(s)

    return {
        "question_id": final["question_id"],
        "surah_id": surah_id,
        "surah_name": surah["surah_name"],
        "ayat_soal": final["soal"],
        "pilihan": final["choices"],
        "correct_index": final["correct_index"],
        "correct_ayat": final["correct"],
        "ai_source": final.get("ai_source", "fallback"),
    }

# -------------------- API --------------------
SESSIONS: Dict[str, dict] = {}

@app.get("/api/surahs", response_model=List[SurahInfo])
def list_surahs():
    return [
        SurahInfo(
            surah_id=s["surah_id"],
            surah_name=s["surah_name"],
            surah_name_ar=s["surah_name_ar"],
            total_ayat=len(s["ayat"]),
        )
        for s in QURAN_DATA
        if len(get_valid_soal(s)) >= 1
    ]

@app.get("/api/question/{surah_id}", response_model=QuestionResponse)
def get_question(surah_id: int):
    q = run_langgraph_pipeline(surah_id)
    if not q:
        raise HTTPException(400, "Tidak ada soal valid.")
    SESSIONS[q["question_id"]] = {
        "correct_index": q["correct_index"],
        "correct_ayat": q["correct_ayat"],
    }
    return QuestionResponse(
        question_id=q["question_id"],
        surah_id=q["surah_id"],
        surah_name=q["surah_name"],
        ayat_soal=AyatOut(**q["ayat_soal"]),
        pilihan=[AyatOut(**p) for p in q["pilihan"]],
        correct_index=q["correct_index"],
        ai_source=q["ai_source"],
    )

@app.post("/api/answer", response_model=AnswerResponse)
def submit_answer(req: AnswerRequest):
    sess = SESSIONS.get(req.question_id)
    if not sess:
        raise HTTPException(400, "Question expired.")
    ok = req.selected_index == sess["correct_index"]
    ca = sess["correct_ayat"]
    expl = f"Benar\n{ca['arab']}\n{ca['id']}" if ok else f"Salah\nYang benar: {ca['arab']}\n{ca['id']}"
    del SESSIONS[req.question_id]
    return AnswerResponse(correct=ok, explanation=expl, next_ready=True)

@app.get("/api/health")
def health():
    ollama_ok = False
    try:
        with httpx.Client(timeout=3.0) as c:
            ollama_ok = c.get(f"{OLLAMA_BASE}/api/tags").status_code == 200
    except Exception:
        pass
    langgraph_ok = False
    try:
        import langgraph  # noqa
        langgraph_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "surah_count": len(QURAN_DATA),
        "ayat_count": sum(len(s["ayat"]) for s in QURAN_DATA),
        "chroma_ok": CHROMA_OK,
        "ollama_ok": ollama_ok,
        "gemini_configured": bool(GEMINI_API_KEY),
        "langgraph_installed": langgraph_ok,
        "pipeline": "LangGraph: retrieve → select(Ollama|Gemini|rule) → format",
        "ai_source_hybrid": "Ollama primary → Gemini fallback → rule last",
    }

ASSETS = BASE / "assets"
FRONTEND = BASE / "frontend"
if ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
