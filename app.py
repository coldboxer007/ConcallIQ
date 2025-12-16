# Streamlit app for processing earnings call transcripts with Gemini.
import os
import re
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import json

import pdfplumber
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Environment & model bootstrap -------------------------------------------------

# Load environment variables from .env and verify Gemini API key availability.
load_dotenv()

# Try Streamlit secrets first (for deployment), then fall back to .env (for local dev)
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
except (FileNotFoundError, KeyError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "⚠️ Google Gemini API key not found. "
        "For local development, ensure your .env file defines GOOGLE_API_KEY. "
        "For Streamlit Cloud deployment, add GOOGLE_API_KEY to your app secrets."
    )
    st.stop()

# Set the API key as an environment variable for google-generativeai library
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Instantiate shared embedding and LLM configurations for reuse across the app.
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)
SUMMARY_LLM = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)
QUESTION_GENERATOR = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.4,
    google_api_key=GOOGLE_API_KEY
)
CLASSIFIER_LLM = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)
QA_LLM = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)


@dataclass
# --- Data containers for retrieval bookkeeping ------------------------------------
class QuestionEntry:
    question: str
    chunk_index: int


@dataclass
class RetrievalConfig:
    questions_per_chunk: int
    top_k_questions: int
    top_k_chunks: int
    window_size: int
    max_segments: int
    show_debug_panels: bool


@dataclass
class QuimIndex:
    chunk_texts: List[str]
    question_entries: List[QuestionEntry]
    question_embeddings: np.ndarray

    def retrieve_chunks(
        self,
        query: str,
        top_k_questions: int,
        top_k_chunks: int,
    ) -> Tuple[List[str], List[Tuple[QuestionEntry, float]]]:
        if not self.question_entries:
            return [], []

        query_vec = np.array(EMBEDDINGS.embed_query(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) + 1e-12
        query_vec /= query_norm

        scores = self.question_embeddings @ query_vec
        ranked_question_indices = np.argsort(scores)[::-1][:top_k_questions]

        selected_chunks: List[str] = []
        seen_indexes: set[int] = set()
        debug_matches: List[Tuple[QuestionEntry, float]] = []
        for idx in ranked_question_indices:
            chunk_idx = self.question_entries[idx].chunk_index
            debug_matches.append((self.question_entries[idx], float(scores[idx])))
            if chunk_idx not in seen_indexes:
                selected_chunks.append(self.chunk_texts[chunk_idx])
                seen_indexes.add(chunk_idx)
            if len(selected_chunks) >= top_k_chunks:
                break
        return selected_chunks, debug_matches


def process_document(uploaded_file) -> str | None:
    # Read the PDF transcript into raw text before any downstream processing.
    """Extract text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        document_text = "\n".join(filter(None, pages)).strip()
        if not document_text:
            st.warning("The uploaded PDF does not contain extractable text.")
            return None
        return document_text
    except Exception as exc:  # noqa: BLE001 - surfaced to UI
        st.error(f"Error processing document: {exc}")
        return None


def split_document_text(document_text: str) -> List[str]:
    """Chunk long documents into overlapping segments for downstream tasks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(document_text)


# --- QuIM question generation helpers --------------------------------------------
def _extract_json_array(text: str) -> List[str]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
    return []


def generate_potential_questions(chunk_text: str, num_questions: int) -> List[str]:
    """Use Gemini to propose potential user questions for a chunk."""
    if not chunk_text.strip():
        return []

    prompt = (
        "You are helping build a question retriever for an earnings call transcript. "
        f"Read the passage and produce {num_questions} distinct, concise questions a financial analyst "
        "might ask to retrieve this passage. Return only a JSON array of strings with no additional commentary.\n\n"
        "Passage:\n" + chunk_text.strip()
    )
    try:
        response = QUESTION_GENERATOR.predict(prompt)
    except Exception:
        return []

    questions = _extract_json_array(response)
    return questions[:num_questions]


# --- QuIM indexing ----------------------------------------------------------------
def build_quim_index(chunks: List[str], questions_per_chunk: int) -> QuimIndex:
    """Construct a question-to-chunk index for QuIM-RAG retrieval."""
    question_entries: List[QuestionEntry] = []
    normalized_embeddings: List[np.ndarray] = []

    for idx, chunk in enumerate(chunks):
        questions = generate_potential_questions(chunk, questions_per_chunk)
        for question in questions:
            question_entries.append(QuestionEntry(question=question, chunk_index=idx))
            vec = np.array(EMBEDDINGS.embed_query(question), dtype=np.float32)
            norm = np.linalg.norm(vec) + 1e-12
            normalized_embeddings.append(vec / norm)

    if normalized_embeddings:
        embedding_matrix = np.vstack(normalized_embeddings)
    else:
        embedding_dim = len(EMBEDDINGS.embed_query("placeholder"))
        embedding_matrix = np.zeros((0, embedding_dim), dtype=np.float32)

    return QuimIndex(
        chunk_texts=chunks,
        question_entries=question_entries,
        question_embeddings=embedding_matrix,
    )


def summarize_document(document_text: str) -> str | None:
    """Generate a high-level summary of the uploaded document."""
    try:
        chunks = split_document_text(document_text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        chain = load_summarize_chain(
            SUMMARY_LLM,
            chain_type="map_reduce",
        )
        return chain.run(docs)
    except Exception as exc:  # noqa: BLE001 - surfaced to UI
        st.error(f"Error summarizing document: {exc}")
        return None


# --- Streamlit render helpers -----------------------------------------------------
def display_summary(summary: str) -> None:
    """Render the generated summary and offer a download option."""
    st.markdown("### 📊 Executive Summary")
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0c4a6e 0%, #075985 100%);
                    padding: 1.5rem;
                    border-radius: 10px;
                    border-left: 4px solid #06b6d4;
                    margin: 1rem 0;
                    color: #ffffff;
                    line-height: 1.8;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            {summary}
        </div>
    """, unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name="concalliq_summary.txt",
        mime="text/plain",
    )


def compute_question_suggestions(
    quim_index: QuimIndex,
    mode: str,
    count: int = 4,
    chunk_index: int | None = None,
) -> List[str]:
    if quim_index is None or not quim_index.question_entries:
        return []

    entries = quim_index.question_entries
    if mode == "First 4":
        return [entry.question for entry in entries[:count]]

    if mode == "Random 4":
        if len(entries) <= count:
            return [entry.question for entry in entries]
        sampled_entries = random.sample(entries, count)
        return [entry.question for entry in sampled_entries]

    if mode == "Chunk focus":
        if chunk_index is None:
            return []
        chunk_entries = [entry for entry in entries if entry.chunk_index == chunk_index]
        return [entry.question for entry in chunk_entries[:count]]

    return []


def display_quim_overview(quim_index: QuimIndex, config: RetrievalConfig) -> None:
    st.markdown("### 🔍 Retrieval Index Analytics")
    total_chunks = len(quim_index.chunk_texts)
    total_questions = len(quim_index.question_entries)
    avg_questions = total_questions / total_chunks if total_chunks else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks indexed", total_chunks)
    col2.metric("Generated questions", total_questions)
    col3.metric("Avg questions/chunk", f"{avg_questions:.1f}")

    if config.show_debug_panels and total_questions:
        sample_size = min(10, total_questions)
        sample_entries = quim_index.question_entries[:sample_size]
        st.markdown("**Sample generated questions**")
        st.dataframe(
            {
                "Question": [entry.question for entry in sample_entries],
                "Chunk #": [entry.chunk_index + 1 for entry in sample_entries],
            },
            use_container_width=True,
        )


# --- Retrieval decision & context selection --------------------------------------
def should_skip_retrieval(question: str) -> Tuple[bool, str]:
    """Decide whether Gemini likely knows the answer without retrieval (FIT-RAG self-knowledge)."""
    classifier_prompt = f"""
You determine if external document retrieval is required to answer a question.
Question: "{question.strip()}"

Respond with exactly one of the following strings:
- NO_RETRIEVE | <short reason>
- RETRIEVE | <short reason>
"""
    try:
        decision = CLASSIFIER_LLM.predict(classifier_prompt)
    except Exception:
        return False, "Classifier error"

    decision_text = decision.strip()
    label, reason = "", ""
    if "|" in decision_text:
        parts = decision_text.split("|", maxsplit=1)
        label = parts[0].strip().upper()
        reason = parts[1].strip()
    else:
        label = decision_text.strip().upper()
        reason = ""

    should_skip = label.startswith("NO_RETRIEVE")
    if not reason:
        reason = "LLM suggested " + ("no retrieval" if should_skip else "retrieval")
    return should_skip, reason


def _sentence_windows(text: str, window_size: int) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    windows: List[str] = []
    for start in range(0, len(sentences) - window_size + 1):
        windows.append(" ".join(sentences[start : start + window_size]))
    return windows


def reduce_subdocuments(
    question: str,
    chunks: Sequence[str],
    window_size: int,
    max_segments: int,
) -> List[str]:
    """Apply FIT-RAG token reduction by selecting the most relevant sub-document windows."""
    candidate_segments: List[str] = []
    for chunk in chunks:
        candidate_segments.extend(_sentence_windows(chunk, window_size))

    candidate_segments = [seg for seg in candidate_segments if seg]
    if not candidate_segments:
        return []

    segment_embeddings = EMBEDDINGS.embed_documents(candidate_segments)
    question_vec = np.array(EMBEDDINGS.embed_query(question), dtype=np.float32)
    question_vec /= np.linalg.norm(question_vec) + 1e-12

    scores = []
    for embedding, segment in zip(segment_embeddings, candidate_segments):
        vec = np.array(embedding, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        scores.append((float(np.dot(vec, question_vec)), segment))

    scores.sort(key=lambda item: item[0], reverse=True)
    top_segments = [segment for _, segment in scores[:max_segments]]
    return top_segments


def build_comprehensive_prompt(
    question: str,
    context_segments: Sequence[str],
    chat_history: Sequence[tuple[str, str]],
    retrieval_used: bool,
) -> str:
    history_lines = [f"{speaker.capitalize()}: {content}" for speaker, content in chat_history[-6:]]
    history_block = "\n".join(history_lines) if history_lines else "None"

    if context_segments:
        passages = "\n\n".join(
            f"Passage {idx + 1} (evidence):\n{segment}" for idx, segment in enumerate(context_segments)
        )
    else:
        passages = "No external passages were retrieved. Rely on trusted internal knowledge." if not retrieval_used else "Relevant passages were not found for this question."

    instructions = """
Instructions:
1. Study the question and all provided passages carefully.
2. When passages are available, ground your answer in them and cite the passage numbers.
3. If no passages are available or they do not contain the answer, explain that fact clearly.
4. Provide a concise answer followed by a brief explanation of how you arrived at it.
"""

    prompt = f"""
You are an earnings call analyst chatbot assisting with financial insights.

Conversation history:
{history_block}

Question:
{question.strip()}

Context Passages:
{passages}

{instructions.strip()}
"""
    return prompt


# --- Question answering orchestration --------------------------------------------
def answer_question(
    question: str,
    quim_index: QuimIndex | None,
    chat_history: Sequence[tuple[str, str]],
    config: RetrievalConfig,
) -> dict:
    """End-to-end question answering with QuIM-RAG and FIT-RAG enhancements."""
    skip_retrieval, skip_reason = should_skip_retrieval(question)
    if skip_retrieval or quim_index is None:
        prompt = build_comprehensive_prompt(question, [], chat_history, retrieval_used=False)
        answer_text = QA_LLM.predict(prompt)
        return {
            "answer": answer_text,
            "strategy": "no_retrieve",
            "segments": [],
            "decision_reason": skip_reason,
            "question_debug": [],
            "token_stats": None,
        }

    candidate_chunks, question_debug = quim_index.retrieve_chunks(
        question,
        top_k_questions=config.top_k_questions,
        top_k_chunks=config.top_k_chunks,
    )
    if not candidate_chunks:
        prompt = build_comprehensive_prompt(question, [], chat_history, retrieval_used=True)
        answer_text = QA_LLM.predict(prompt)
        return {
            "answer": answer_text,
            "strategy": "no_context",
            "segments": [],
            "decision_reason": "No matching indexed questions",
            "question_debug": question_debug,
            "token_stats": None,
        }

    reduced_segments = reduce_subdocuments(
        question,
        candidate_chunks,
        window_size=config.window_size,
        max_segments=config.max_segments,
    )
    if not reduced_segments:
        reduced_segments = candidate_chunks

    prompt = build_comprehensive_prompt(question, reduced_segments, chat_history, retrieval_used=True)
    answer_text = QA_LLM.predict(prompt)

    original_tokens = sum(len(chunk) for chunk in candidate_chunks) / 4
    reduced_tokens = sum(len(segment) for segment in reduced_segments) / 4
    savings_pct = 0.0
    if original_tokens:
        savings_pct = max(0.0, 1 - (reduced_tokens / original_tokens)) * 100

    token_stats = {
        "original_tokens_est": round(original_tokens, 1),
        "reduced_tokens_est": round(reduced_tokens, 1),
        "savings_pct": round(savings_pct, 1),
    }

    return {
        "answer": answer_text,
        "strategy": "rag",
        "segments": reduced_segments,
        "decision_reason": skip_reason,
        "question_debug": question_debug,
        "token_stats": token_stats,
    }


# --- Streamlit entry point -------------------------------------------------------
def run_app() -> None:
    """Launch the Streamlit interface for the earnings analysis workflow."""
    st.set_page_config(
        page_title="ConcallIQ: Intelligent Financial Document Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for professional styling
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #1e3a8a;
            --secondary-color: #0891b2;
            --accent-color: #06b6d4;
            --success-color: #059669;
            --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }

        .main-header p {
            color: #e0f2fe;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            margin-bottom: 0;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #ffffff;
            font-weight: 600;
        }

        /* Sidebar labels and text */
        [data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
            font-weight: 500;
        }

        [data-testid="stSidebar"] .stMarkdown p {
            color: #cbd5e1 !important;
        }

        /* Sidebar slider values */
        [data-testid="stSidebar"] .stSlider {
            color: #e2e8f0;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e3a8a;
        }

        [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 500;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        }

        /* Input fields */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            padding: 0.6rem;
            transition: border-color 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #0891b2;
            box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.1);
        }

        .stTextInput label {
            color: #1e293b !important;
            font-weight: 500;
        }

        /* File uploader - Blue gradient theme */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
            border: 2px dashed #60a5fa;
            border-radius: 10px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #93c5fd;
            background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }

        [data-testid="stFileUploader"] label {
            color: #ffffff !important;
            font-weight: 600;
        }

        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] small {
            color: #e0f2fe !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-weight: 600;
            color: #1e3a8a;
        }

        .streamlit-expanderHeader:hover {
            background: #f0f9ff;
            border-color: #0891b2;
        }

        /* Info/success boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }

        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        }

        /* Subheader styling */
        .stMarkdown h2 {
            color: #1e3a8a;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        .stMarkdown h3 {
            color: #0891b2;
            font-weight: 600;
        }

        /* Radio buttons */
        .stRadio > label {
            font-weight: 600;
            color: #1e293b !important;
        }

        .stRadio div[role="radiogroup"] label {
            color: #334155 !important;
        }

        /* Checkbox */
        .stCheckbox label {
            color: #1e293b !important;
            font-weight: 500;
        }

        /* Slider styling */
        .stSlider > div > div > div {
            background: #e2e8f0;
        }

        .stSlider label {
            color: #1e293b !important;
            font-weight: 500;
        }

        /* Sidebar slider override for dark theme */
        [data-testid="stSidebar"] .stSlider label {
            color: #e2e8f0 !important;
        }

        /* Selectbox styling */
        .stSelectbox label {
            color: #1e293b !important;
            font-weight: 500;
        }

        /* Sidebar selectbox override for dark theme */
        [data-testid="stSidebar"] .stSelectbox label {
            color: #e2e8f0 !important;
        }

        /* Sidebar checkbox override for dark theme */
        [data-testid="stSidebar"] .stCheckbox label {
            color: #e2e8f0 !important;
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #0891b2 !important;
        }

        /* Container styling */
        .element-container {
            margin-bottom: 1rem;
        }

        /* Caption styling */
        .stCaption {
            color: #64748b;
            font-style: italic;
        }

        /* Table styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Suggestion buttons grid */
        .suggestion-buttons {
            display: grid;
            gap: 0.5rem;
        }

        /* Global label color fix for visibility */
        label {
            color: #1e293b;
        }

        /* Main content area text */
        .main .block-container {
            color: #1e293b;
        }

        /* Answer box - force all text to white */
        .stMarkdown p span,
        .stMarkdown p strong,
        .stMarkdown p em,
        .stMarkdown p b,
        .stMarkdown p i,
        .stMarkdown p code {
            color: inherit !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # App header
    st.markdown("""
        <div class="main-header">
            <h1>🧠 ConcallIQ</h1>
            <p>Intelligent Financial Document Summarization and Q&A System</p>
        </div>
    """, unsafe_allow_html=True)

    # Prime session state with the structures the app expects on first load.
    if "quim_index" not in st.session_state:
        st.session_state.quim_index = None
        st.session_state.summary = None
        st.session_state.chat_history = []
    if "user_question_input" not in st.session_state:
        st.session_state.user_question_input = ""
    if "suggestion_mode" not in st.session_state:
        st.session_state.suggestion_mode = "First 4"
    if "suggestion_chunk_idx" not in st.session_state:
        st.session_state.suggestion_chunk_idx = 0
    if "reset_user_question_input" not in st.session_state:
        st.session_state.reset_user_question_input = False

    # Pull the user's previously-selected retrieval parameters, or fallback to defaults.
    default_config = st.session_state.get(
        "retrieval_config",
        RetrievalConfig(
            questions_per_chunk=3,
            top_k_questions=12,
            top_k_chunks=4,
            window_size=3,
            max_segments=6,
            show_debug_panels=True,
        ),
    )

    # Sidebar exposes controls for how aggressive retrieval and compression should be.
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Controls")
        st.caption("Fine-tune retrieval and compression parameters")
        questions_per_chunk = st.slider(
            "Questions per chunk",
            min_value=1,
            max_value=6,
            value=default_config.questions_per_chunk,
        )
        top_k_questions = st.slider(
            "Top question matches",
            min_value=4,
            max_value=30,
            value=default_config.top_k_questions,
        )
        top_k_chunks = st.slider(
            "Max supporting chunks",
            min_value=1,
            max_value=8,
            value=default_config.top_k_chunks,
        )
        window_size = st.slider(
            "Sentence window size",
            min_value=1,
            max_value=5,
            value=default_config.window_size,
        )
        max_segments = st.slider(
            "Max sub-doc segments",
            min_value=2,
            max_value=12,
            value=default_config.max_segments,
        )
        show_debug_panels = st.checkbox(
            "Show debug panels",
            value=default_config.show_debug_panels,
        )

    config = RetrievalConfig(
        questions_per_chunk=questions_per_chunk,
        top_k_questions=top_k_questions,
        top_k_chunks=top_k_chunks,
        window_size=window_size,
        max_segments=max_segments,
        show_debug_panels=show_debug_panels,
    )
    st.session_state.retrieval_config = config

    # Users upload one PDF earnings transcript to kick off the pipeline.
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your earnings call transcript (PDF)",
        type=["pdf"],
        help="Upload a PDF transcript of an earnings call to summarize and query using AI-powered analysis.",
    )
    process_button = st.button("🚀 Process Document", type="primary")

    if process_button:
        if not uploaded_file:
            st.warning("Please upload a PDF before processing.")
        else:
            # Convert the PDF to text, build retrieval index, and generate an executive summary.
            with st.spinner("Processing document with Gemini..."):
                document_text = process_document(uploaded_file)
                if document_text:
                    chunks = split_document_text(document_text)
                    with st.spinner("Generating QuIM question index..."):
                        quim_index = build_quim_index(chunks, config.questions_per_chunk)
                    st.session_state.quim_index = quim_index
                    st.session_state.chat_history = []
                    summary = summarize_document(document_text)
                    st.session_state.summary = summary
                    if summary:
                        st.success("Document processed successfully!")

    if st.session_state.quim_index:
        display_quim_overview(st.session_state.quim_index, config)

    if st.session_state.summary:
        display_summary(st.session_state.summary)

    st.divider()
    st.markdown("### 💬 Intelligent Q&A Interface")
    st.caption("Ask questions about your earnings call transcript using advanced RAG techniques")

    suggestion_container = st.container()
    with suggestion_container:
        if st.session_state.quim_index is None:
            st.info("Upload and process a document to receive suggested prompts.")
            suggestions: List[str] = []
            selected_chunk_index: int | None = None
        elif not st.session_state.quim_index.question_entries:
            st.info("Question suggestions will appear once indexing completes.")
            suggestions = []
            selected_chunk_index = None
        else:
            suggestions = []
            selected_chunk_index = None
            current_mode = st.session_state.get("suggestion_mode", "First 4")
            mode_index = {"First 4": 0, "Random 4": 1, "Chunk focus": 2}.get(current_mode, 0)
            # Let analysts toggle how suggestions are sampled from the synthetic question bank.
            suggestion_mode = st.radio(
                "Suggestion mode",
                ("First 4", "Random 4", "Chunk focus"),
                index=mode_index,
                key="suggestion_mode",
            )

            if suggestion_mode == "Chunk focus":
                total_chunks = len(st.session_state.quim_index.chunk_texts)
                if st.session_state.suggestion_chunk_idx >= total_chunks:
                    st.session_state.suggestion_chunk_idx = 0
                chunk_options = list(range(total_chunks))
                selected_chunk_index = st.selectbox(
                    "Chunk to explore",
                    options=chunk_options,
                    format_func=lambda idx: f"Chunk {idx + 1}",
                    key="suggestion_chunk_idx",
                )
            else:
                selected_chunk_index = None

            # Slice the question bank according to the user's selection.
            suggestions = compute_question_suggestions(
                st.session_state.quim_index,
                suggestion_mode,
                count=4,
                chunk_index=selected_chunk_index,
            )

            if suggestions:
                st.caption("Suggested prompts (click to populate the question box):")
                columns = st.columns(min(len(suggestions), 2))
                for idx, suggestion in enumerate(suggestions):
                    col = columns[idx % len(columns)]
                    if col.button(
                        suggestion,
                        key=f"suggestion_btn_{suggestion_mode}_{selected_chunk_index}_{idx}",
                    ):
                        st.session_state["user_question_input"] = suggestion
                        st.session_state["reset_user_question_input"] = False
            else:
                st.info("No suggestions available for the current selection.")

    # Clear the input box on the render that follows a successful answer generation.
    if st.session_state.reset_user_question_input:
        st.session_state["user_question_input"] = ""
        st.session_state["reset_user_question_input"] = False

    user_input = st.text_input(
        "💭 Ask a question about the earnings call:",
        placeholder="e.g., What were the CEO's main concerns for next quarter?",
        key="user_question_input",
    )
    if st.button("✨ Get AI Response"):
        question_to_ask = user_input.strip()
        if not question_to_ask:
            st.warning("Please provide a question before requesting a response.")
        elif st.session_state.quim_index is None:
            st.warning("Process a document first to enable the chatbot.")
        else:
            # Run the hybrid QuIM + FIT pipeline and surface the answer together with diagnostics.
            with st.spinner("Generating response..."):
                result = answer_question(
                    question_to_ask,
                    st.session_state.quim_index,
                    st.session_state.chat_history,
                    config,
                )

            st.session_state.chat_history.extend(
                [("user", question_to_ask), ("assistant", result["answer"])]
            )
            st.session_state["reset_user_question_input"] = True

            strategy_label = {
                "no_retrieve": "🎯 Direct Answer (No Retrieval)",
                "no_context": "⚠️ No Relevant Context Found",
                "rag": "✅ Context-Aware RAG Answer",
            }.get(result["strategy"], "Answer")

            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                            padding: 1.5rem;
                            border-radius: 10px;
                            border-left: 4px solid #60a5fa;
                            margin: 1rem 0;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            color: #ffffff;">
                    <strong style="color: #ffffff; font-size: 1.1rem;">🤖 ConcallIQ — {strategy_label}</strong>
                    <p style="color: #ffffff; margin-top: 0.8rem; line-height: 1.8; font-size: 1rem;">
                        <span style="color: #ffffff !important;">{result['answer']}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.caption(f"Retrieval decision: {result['decision_reason']}")

            if result["token_stats"]:
                # Surface how much context was trimmed away by the FIT reducer.
                tokens = result["token_stats"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Original context tokens (est)", tokens["original_tokens_est"])
                col2.metric("Reduced tokens (est)", tokens["reduced_tokens_est"])
                col3.metric("Savings %", f"{tokens['savings_pct']}%")

            if result["segments"]:
                # Reveal the passages we fed back into Gemini for traceability.
                with st.expander("📄 View Context Passages"):
                    for idx, segment in enumerate(result["segments"], start=1):
                        st.markdown(f"**Passage {idx}:**\n{segment}")

            if config.show_debug_panels and result["question_debug"]:
                # Display the synthetic questions and similarity scores that triggered retrieval.
                with st.expander("🔎 Matched Synthetic Questions (Debug)"):
                    st.table(
                        {
                            "Question": [entry.question for entry, _ in result["question_debug"]],
                            "Score": [f"{score:.3f}" for _, score in result["question_debug"]],
                            "Chunk #": [entry.chunk_index + 1 for entry, _ in result["question_debug"]],
                        }
                    )

    if st.session_state.chat_history and config.show_debug_panels:
        # Keep the recent back-and-forth handy when debugging conversation flow.
        with st.expander("💬 Conversation History"):
            for speaker, content in st.session_state.chat_history[-10:]:
                st.markdown(f"**{speaker.capitalize()}:** {content}")


if __name__ == "__main__":
    # Allow the script to be run directly (useful for local debugging outside `streamlit run`).
    run_app()