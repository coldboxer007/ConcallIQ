# 🧠 ConcallIQ: Intelligent Financial Document Summarization and Q&A System

A modernized Streamlit experience powered by Google Gemini for exploring earnings call transcripts end-to-end. Upload a PDF, generate an executive summary, and interrogate the document with an intelligent retrieval-augmented chatbot that showcases **QuIM-RAG question indexing**, **FIT-RAG self-knowledge & token compression**, and **rich UI diagnostics** in real time.

---

## 📌 Overview

| Topic | Details |
| --- | --- |
| **Primary goal** | Help analysts ingest transcripts, surface highlights, and ask follow-up questions without leaving the browser. |
| **Language model** | Gemini “gemini-flash-lite-latest” for summarization, question synthesis, decisioning, and answers. |
| **Embedding model** | `models/text-embedding-004` via `GoogleGenerativeAIEmbeddings`. |
| **Data ingress** | Local PDF upload (parsed with `pdfplumber` into clean text). |
| **Retrieval stack** | QuIM-style synthetic questions + cosine search, followed by FIT-RAG decisioning and token reduction. |
| **UI framework** | Streamlit with live sliders, debug tables, and suggestion buttons. |

---

## 🆕 ConcallIQ Key Features

| Area | Capability |
| --- | --- |
| **LLM Provider** | Google Gemini across all stages (embeddings, summarization, QA, classification). |
| **Retrieval System** | QuIM in-memory index built from AI-generated synthetic questions. |
| **Advanced RAG** | QuIM synthetic question search + FIT-RAG token compression with top-k chunk selection. |
| **Smart Decisioning** | FIT-RAG self-knowledge gate with automatic retrieval strategy selection. |
| **Professional UI** | Modern gradient design, sidebar controls, prompt suggestions, token metrics, debug panels. |
| **Full Observability** | Retrieval decision reasoning, matched question tables, context passages, token savings %, conversation history. |

ConcallIQ is an explainable, configurable, and efficient financial analyst assistant that shows *why* a chunk was selected, *how much* context was consumed, and *which* prompts analysts might ask next—all with a beautiful, professional interface.

---

## ⚙️ Architecture at a Glance

1. **Document ingestion** – `process_document` extracts raw text from PDF (pdfplumber) and stores it in memory.
2. **Chunking** – `split_document_text` applies a recursive character splitter (1 000 characters, 150 overlap) for consistent segments.
3. **Question synthesis** – `generate_potential_questions` asks Gemini to create analyst-style prompts per chunk, feeding `_extract_json_array` for parsing.
4. **QuIM index** – `build_quim_index` embeds each synthetic question, normalizes vectors, and keeps `(question, chunk_index)` metadata for quick retrieval.
5. **Retrieval decision** – `should_skip_retrieval` (FIT step 1) lets Gemini decide if the question can be answered from model knowledge alone.
6. **Context selection** – `QuimIndex.retrieve_chunks` finds chunk candidates, then `reduce_subdocuments` (FIT step 2) pulls sentence windows and ranks them via cosine similarity, reporting estimated tokens saved.
7. **Prompt construction** – `build_comprehensive_prompt` blends reduced context with conversation memory.
8. **Answering & telemetry** – `answer_question` returns the answer alongside strategy, matched questions, token stats, and segments for the Streamlit UI to render.

---

## 🚀 Real-Time Interaction Flow

1. **Upload** a PDF earnings transcript.
2. **Process Document** – the app extracts text, builds the QuIM index, and generates a summary (with spinner feedback for each stage).
3. **Review Summary** – read or download the map-reduce synopsis from Gemini.
4. **Tune Retrieval Controls** – use sidebar sliders to adjust question density, retrieval breadth, and FIT segment limits.
5. **Leverage Suggested Prompts** – choose from *First 4*, *Random 4*, or *Chunk focus* question sets to seed the chat box.
6. **Ask a Question** – the app logs the conversation history, reasons about whether to retrieve, and streams back an answer.
7. **Inspect Diagnostics** – drill into matched questions, context passages, and token savings to understand the response provenance.

---

## 🔍 Feature Deep Dive

### QuIM-RAG Question Index
- Gemini generates analyst-style questions per chunk (`questions_per_chunk` slider).
- Questions are embedded and normalized to enable cosine similarity search.
- Retrieval returns both the chunk text and the driving synthetic questions, allowing the UI to display *why* a passage was selected.

### FIT-RAG Enhancements
- **Self-knowledge gate**: `should_skip_retrieval` prompts Gemini to return `NO_RETRIEVE` or `RETRIEVE`, providing a rationale (surfaced as “Retrieval decision”).
- **Sentence-window reduction**: `reduce_subdocuments` slices chunks into `window_size` sentences, ranks them, and caps the number via `max_segments`.
- **Token telemetry**: estimated original vs reduced tokens and savings percentage appear when context is used.

### Prompt Suggestions & UI Controls
- Radio buttons toggle suggestion generation strategy; chunk focus mode exposes a select box enumerating all processed chunks.
- Clicking a suggestion pre-fills the chat input without triggering a new rerun, thanks to session-state guards.
- Sidebar sliders provide real-time control over QuIM and FIT thresholds without restarting the app.

### Debugging & Observability
- **Matched synthetic questions** expander lists the QuIM hits with similarity scores and chunk numbers.
- **Context passages** expander shows the exact snippets passed to Gemini for the answer.
- **Conversation history** expander keeps recent exchanges handy for troubleshooting multi-turn threads.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Google AI Studio](https://ai.google.dev/) API key with access to Gemini models
- (Optional) `virtualenv`, `pyenv`, or Conda for environment isolation

### Installation
```bash
git clone https://github.com/KaifAhmad1/Earning-Analysis-Application.git
cd Earning-Analysis-Application  # or your local directory name
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY="your-gemini-api-key"
```

### Run the App

```bash
streamlit run app.py
```

Open the URL printed to the terminal (usually `http://localhost:8501`).

---

## 🚀 Deploying to Streamlit Cloud

### Prerequisites
- GitHub account
- Google Gemini API key
- Fork or push this repository to GitHub

### Deployment Steps

1. **Push your code to GitHub** (ensure `.env` is in `.gitignore`)
   ```bash
   git add .
   git commit -m "Prepare ConcallIQ for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`

3. **Configure Secrets**
   - In your Streamlit Cloud app settings, go to "Secrets"
   - Add your Google API key:
     ```toml
     GOOGLE_API_KEY = "your-actual-api-key-here"
     ```

4. **Deploy!**
   - Click "Deploy"
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### Security Notes
- Never commit your `.env` file or API keys to GitHub
- Use Streamlit Cloud secrets for production deployment
- The app automatically detects if it's running locally (.env) or on Streamlit Cloud (secrets)

---

## 🛠️ Configuration Quick Reference

| Control | Location | Description |
| --- | --- | --- |
| `questions_per_chunk` | Sidebar slider | How many synthetic questions Gemini generates per transcript chunk. |
| `top_k_questions` | Sidebar slider | Maximum number of QuIM questions considered per user query. |
| `top_k_chunks` | Sidebar slider | Upper bound on the number of unique chunks fed to FIT. |
| `window_size` | Sidebar slider | Number of sentences per sliding window when trimming context. |
| `max_segments` | Sidebar slider | Total sentence windows retained after scoring. |
| `show_debug_panels` | Sidebar checkbox | Toggles the matched questions, context passages, and conversation history expanders. |
| Suggestion mode | Main panel radio | Chooses First 4, Random 4, or Chunk focus suggestions. |

---

## 🧪 Using ConcallIQ Day-to-Day

1. **Process a fresh transcript** – Every upload resets the summary, index, and chat history to avoid stale context.
2. **Skim the executive summary** – Capture high-level highlights before diving into detailed Q&A.
3. **Customize retrieval parameters** – Tighten sliders for targeted questions or widen them for exploratory sessions.
4. **Use AI-generated suggestions** – Test the dataset quality quickly by clicking a pre-generated prompt.
5. **Ask intelligent follow-ups** – Explore qualitative guidance ("What risks did the CFO emphasize?") or quantitative metrics ("How did revenue guidance change?").
6. **Audit the answer provenance** – Use the context passages and matched questions to verify source alignment and understand the AI's reasoning.

---

## 🧯 Troubleshooting & Tips

- **Empty or malformed PDFs**: The app warns if no text could be extracted. Double-check the source file or convert to OCR text first.
- **Gemini rate limits**: Heavy use of question generation can consume tokens quickly; reuse the same transcript during a session to avoid regenerating the index.
- **Deprecation warnings**: LangChain currently warns about `predict`/`run`; migration to `invoke` is on the roadmap.
- **Torch class warnings**: Some local PyTorch installations emit `_path` warnings when importing embeddings; they are benign but can be silenced by reinstalling torch.
- **Session resets**: Processing a new document clears previous chat history to prevent cross-talk between transcripts.

---

## 🗺️ ConcallIQ Roadmap

- **Async Streaming**: Switch Gemini calls to the new `invoke` API with async streaming for real-time responses.
- **Enhanced PDF Support**: Explore Gemini-native PDF ingestion for scanned documents and complex layouts.
- **Index Persistence**: Persist QuIM indices to disk for instant reloads across sessions.
- **Testing Suite**: Add comprehensive unit and integration tests for retrieval pipeline and UI state management.
- **Multi-Document Support**: Enable simultaneous analysis of multiple earnings call transcripts.
- **Export Features**: Advanced export options including PDF reports, structured JSON, and visualization dashboards.

---

## 🤝 Contributing

We welcome issues and pull requests! If you're planning significant architecture changes (e.g., multi-document support, dashboard embedding, or new RAG techniques), please open a discussion first so we can coordinate the direction and ensure alignment with ConcallIQ's vision.

---

## 📜 License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
