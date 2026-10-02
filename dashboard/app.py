import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.csv"
PDF_PATH = PROJECT_ROOT / "data" / "company_report.pdf"


# ============================================================
# PYTHON PATH
# ============================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: #6b7280;
        margin-bottom: 1.3rem;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #ffffff;
        min-height: 110px;
     color: #111827 !important;}

    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .metric-description {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    .pipeline-box {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem;
        background: #fafafa;
        text-align: center;
        min-height: 100px;
     color: #111827 !important;}

    .pipeline-name {
        font-weight: 650;
        font-size: 0.95rem;
    }

    .pipeline-detail {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

@st.cache_data
def load_results():

    if not RESULTS_PATH.exists():
        return None

    data = pd.read_csv(RESULTS_PATH)

    if "retrieved_contexts" in data.columns:

        def parse_contexts(value):

            if isinstance(value, list):
                return value

            if pd.isna(value):
                return []

            try:
                parsed = json.loads(value)

                if isinstance(parsed, list):
                    return parsed

            except Exception:
                pass

            return []

        data["retrieved_contexts"] = (
            data["retrieved_contexts"]
            .apply(parse_contexts)
        )

    return data


df = load_results()


# ============================================================
# VALIDATE RESULTS
# ============================================================

if df is None:

    st.title("RAG Evaluation Dashboard")

    st.write(
        "Evaluation results could not be found."
    )

    st.code(
        str(RESULTS_PATH),
        language="text"
    )

    st.stop()


required_columns = [
    "question",
    "answer",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.title("RAG Evaluation Dashboard")

    st.write(
        "The evaluation CSV is missing these required columns:"
    )

    st.write(
        ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# GLOBAL EVALUATION VALUES
# ============================================================

faithfulness = float(
    df["faithfulness"].mean()
)

answer_relevancy = float(
    df["answer_relevancy"].mean()
)

context_precision = float(
    df["context_precision"].mean()
)

context_recall = float(
    df["context_recall"].mean()
)

# Dashboard-derived unsupported claim / hallucination diagnostic.
# This is NOT an official RAGAS hallucination metric.
hallucination_diagnostic = (
    1.0 - faithfulness
)


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

@st.cache_data
def get_document_statistics():

    try:

        from src.document_loader import load_pdf
        from src.chunker import chunk_text

        text = load_pdf(
            str(PDF_PATH)
        )

        chunks = chunk_text(
            text,
            chunk_size=800,
            overlap=150
        )

        lengths = [
            len(chunk)
            for chunk in chunks
        ]

        return {
            "text_length": len(text),
            "chunks": chunks,
            "lengths": lengths,
            "error": None
        }

    except Exception as error:

        return {
            "text_length": 0,
            "chunks": [],
            "lengths": [],
            "error": str(error)
        }


document_stats = get_document_statistics()

chunk_lengths = document_stats["lengths"]


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

@st.cache_resource
def load_retrieval_pipeline():

    from src.rag_pipeline import RAGPipeline

    return RAGPipeline(
        pdf_path=str(PDF_PATH),
        chunk_size=800,
        overlap=150,
        candidate_k=8,
        top_k=5
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "retrieval_history" not in st.session_state:
    st.session_state.retrieval_history = []

if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(
    label,
    value,
    description=""
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(
    title,
    description
):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True
    )


def pipeline_box(
    name,
    detail
):

    st.markdown(
        f"""
        <div class="pipeline-box">
            <div class="pipeline-name">{name}</div>
            <div class="pipeline-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">RAG Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Analytics and retrieval diagnostics'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select section",
        [
            "Dashboard",
            "Pipeline",
            "Document & Chunking",
            "Evaluation & Hallucination",
            "Answer Inspection",
            "Retrieval Lab",
            "Query Analysis"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown("### Current Configuration")

    st.write("Chunk Size: 800")
    st.write("Chunk Overlap: 150")
    st.write("Candidates: 8")
    st.write("Final Contexts: 5")

    st.divider()

    st.markdown("### Evaluation")

    st.write(
        f"Faithfulness: {faithfulness:.4f}"
    )

    st.write(
        f"Answer Relevancy: {answer_relevancy:.4f}"
    )

    st.write(
        f"Context Precision: {context_precision:.4f}"
    )

    st.write(
        f"Context Recall: {context_recall:.4f}"
    )

    st.divider()

    st.caption(
        "Evaluation data is loaded from "
        "results/evaluation_results.csv."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'RAG Evaluation & Analytics Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Evaluate retrieval quality, answer quality, '
    'groundedness, and RAG pipeline behaviour.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1
# DASHBOARD
# ============================================================

if page == "Dashboard":

    page_header(
        "System Overview",
        "A concise view of how the RAG system is configured "
        "and how it performed during evaluation."
    )

    # --------------------------------------------------------
    # SECTION 1: RAG CONFIGURATION
    # --------------------------------------------------------

    st.subheader("RAG Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "Chunk Size",
            "800",
            "Maximum characters per chunk"
        )

    with col2:

        metric_card(
            "Chunk Overlap",
            "150",
            "Characters shared between chunks"
        )

    with col3:

        metric_card(
            "Document Chunks",
            "1042",
            "Chunks generated from the PDF"
        )

    with col4:

        metric_card(
            "Retrieval Candidates",
            "8",
            "Dense retrieval candidates"
        )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "Final Contexts",
            "5",
            "Contexts passed after reranking"
        )

    with col2:

        metric_card(
            "Embedding Dimension",
            "384",
            "all-MiniLM-L6-v2 vector size"
        )

    with col3:

        metric_card(
            "Step Size",
            "650",
            "800 minus 150 overlap"
        )

    with col4:

        metric_card(
            "Evaluation Questions",
            str(len(df)),
            "Questions evaluated by RAGAS"
        )

    # --------------------------------------------------------
    # SECTION 2: EVALUATION
    # --------------------------------------------------------

    st.subheader("Evaluation Results")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card(
            "Faithfulness",
            f"{faithfulness:.4f}",
            "Claims supported by retrieved context"
        )

    with col2:

        metric_card(
            "Hallucination Diagnostic",
            f"{hallucination_diagnostic:.4f}",
            "1 - Faithfulness"
        )

    with col3:

        metric_card(
            "Answer Relevancy",
            f"{answer_relevancy:.4f}",
            "Answer relevance to the question"
        )

    with col4:

        metric_card(
            "Context Precision",
            f"{context_precision:.4f}",
            "Relevance of retrieved context"
        )

    with col5:

        metric_card(
            "Context Recall",
            f"{context_recall:.4f}",
            "Ground-truth information retrieved"
        )

    st.caption(
        "The Hallucination Diagnostic is a dashboard-derived "
        "unsupported-claim indicator calculated as "
        "1 - Faithfulness. It is not an official RAGAS "
        "hallucination metric."
    )

    # --------------------------------------------------------
    # SECTION 3: METRIC BARS
    # --------------------------------------------------------

    st.subheader("Evaluation Score Overview")

    metrics_for_display = [
        ("Faithfulness", faithfulness),
        ("Answer Relevancy", answer_relevancy),
        ("Context Precision", context_precision),
        ("Context Recall", context_recall)
    ]

    for label, score in metrics_for_display:

        st.write(
            f"{label}: {score:.4f}"
        )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    score
                )
            )
        )

    # --------------------------------------------------------
    # SECTION 4: PIPELINE
    # --------------------------------------------------------

    st.subheader("RAG Pipeline")

    pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = (
        st.columns(4)
    )

    with pipeline_col1:

        pipeline_box(
            "1. Document",
            "NVIDIA Annual Report"
        )

    with pipeline_col2:

        pipeline_box(
            "2. Chunking",
            "800 / 150 overlap"
        )

    with pipeline_col3:

        pipeline_box(
            "3. Embeddings",
            "all-MiniLM-L6-v2"
        )

    with pipeline_col4:

        pipeline_box(
            "4. FAISS",
            "Dense retrieval"
        )

    st.write("")

    pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = (
        st.columns(4)
    )

    with pipeline_col1:

        pipeline_box(
            "5. Retrieval",
            "Top 8 candidates"
        )

    with pipeline_col2:

        pipeline_box(
            "6. Reranking",
            "Cross-Encoder"
        )

    with pipeline_col3:

        pipeline_box(
            "7. Generation",
            "Groq LLM"
        )

    with pipeline_col4:

        pipeline_box(
            "8. Evaluation",
            "RAGAS"
        )

    # --------------------------------------------------------
    # SECTION 5: WHAT EACH SCORE MEANS
    # --------------------------------------------------------

    st.subheader("How to Read This Dashboard")

    meaning_table = pd.DataFrame(
        [
            {
                "Metric": "Faithfulness",
                "Meaning":
                    "Measures whether generated claims are "
                    "supported by the retrieved context."
            },
            {
                "Metric": "Hallucination Diagnostic",
                "Meaning":
                    "Dashboard-derived 1 - Faithfulness indicator. "
                    "Not an official RAGAS hallucination metric."
            },
            {
                "Metric": "Answer Relevancy",
                "Meaning":
                    "Measures how relevant the generated answer "
                    "is to the question."
            },
            {
                "Metric": "Context Precision",
                "Meaning":
                    "Measures how relevant the retrieved context "
                    "is for the question."
            },
            {
                "Metric": "Context Recall",
                "Meaning":
                    "Measures whether relevant information needed "
                    "for the answer was retrieved."
            }
        ]
    )

    st.dataframe(
        meaning_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2
# PIPELINE
# ============================================================

elif page == "Pipeline":

    page_header(
        "RAG Pipeline",
        "Technical configuration of each stage in the RAG system."
    )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.subheader("Core Configuration")

    configuration = pd.DataFrame(
        [
            {
                "Component": "Document",
                "Configuration": "NVIDIA 2025 Annual Report"
            },
            {
                "Component": "Chunk Size",
                "Configuration": "800 characters"
            },
            {
                "Component": "Chunk Overlap",
                "Configuration": "150 characters"
            },
            {
                "Component": "Step Size",
                "Configuration": "650 characters"
            },
            {
                "Component": "Embedding Model",
                "Configuration":
                    "sentence-transformers/all-MiniLM-L6-v2"
            },
            {
                "Component": "Embedding Dimension",
                "Configuration": "384"
            },
            {
                "Component": "Vector Store",
                "Configuration": "FAISS IndexFlatIP"
            },
            {
                "Component": "Candidate Retrieval",
                "Configuration": "8"
            },
            {
                "Component": "Reranker",
                "Configuration":
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
            },
            {
                "Component": "Final Contexts",
                "Configuration": "5"
            },
            {
                "Component": "LLM",
                "Configuration":
                    "openai/gpt-oss-120b"
            },
            {
                "Component": "Provider",
                "Configuration": "Groq"
            },
            {
                "Component": "Temperature",
                "Configuration": "0"
            }
        ]
    )

    st.dataframe(
        configuration,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader("Architecture")

    st.code(
        """
PDF
 |
 v
Document Loader
 |
 v
Chunking
 | 800 characters
 | 150 character overlap
 v
Sentence Transformer
 |
 v
FAISS Vector Store
 |
 v
Dense Retrieval
 | Top 8
 v
Cross-Encoder
 | Top 5
 v
Retrieved Context
 |
 v
Groq LLM
 |
 v
Generated Answer
 |
 v
RAGAS Evaluation
 |-- Faithfulness
 |-- Answer Relevancy
 |-- Context Precision
 +-- Context Recall
""",
        language="text"
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    st.subheader("Chunking Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "Chunk Size",
            "800",
            "Maximum characters"
        )

    with col2:

        metric_card(
            "Chunk Overlap",
            "150",
            "Shared characters"
        )

    with col3:

        metric_card(
            "Step Size",
            "650",
            "800 - 150"
        )

    st.write(
        "Chunk overlap helps preserve information when a "
        "sentence or concept crosses a chunk boundary."
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    st.subheader("Embeddings")

    embedding_table = pd.DataFrame(
        [
            {
                "Property": "Model",
                "Value":
                    "sentence-transformers/all-MiniLM-L6-v2"
            },
            {
                "Property": "Dimension",
                "Value": "384"
            },
            {
                "Property": "Framework",
                "Value": "Sentence Transformers"
            }
        ]
    )

    st.dataframe(
        embedding_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    st.subheader("Vector Store")

    vector_table = pd.DataFrame(
        [
            {
                "Property": "Vector Store",
                "Value": "FAISS"
            },
            {
                "Property": "Index",
                "Value": "IndexFlatIP"
            },
            {
                "Property": "Normalization",
                "Value": "L2 normalization"
            },
            {
                "Property": "Similarity",
                "Value": "Cosine similarity"
            }
        ]
    )

    st.dataframe(
        vector_table,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Because document and query embeddings are normalized, "
        "FAISS inner product is equivalent to cosine similarity."
    )

    # --------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------

    st.subheader("Cross-Encoder Reranker")

    reranker_table = pd.DataFrame(
        [
            {
                "Property": "Model",
                "Value":
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
            },
            {
                "Property": "Input",
                "Value": "Query + Candidate Document"
            },
            {
                "Property": "Candidates",
                "Value": "8"
            },
            {
                "Property": "Final Results",
                "Value": "5"
            },
            {
                "Property": "Score Type",
                "Value": "Raw relevance score"
            }
        ]
    )

    st.dataframe(
        reranker_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    st.subheader("Generation")

    generation_table = pd.DataFrame(
        [
            {
                "Property": "Provider",
                "Value": "Groq"
            },
            {
                "Property": "Model",
                "Value": "openai/gpt-oss-120b"
            },
            {
                "Property": "Temperature",
                "Value": "0"
            },
            {
                "Property": "Strategy",
                "Value": "Context-grounded generation"
            }
        ]
    )

    st.dataframe(
        generation_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3
# DOCUMENT & CHUNKING
# ============================================================

elif page == "Document & Chunking":

    page_header(
        "Document & Chunking",
        "Understand exactly how the source PDF becomes retrieval chunks."
    )

    if not chunk_lengths:

        st.write(
            "Document statistics could not be loaded."
        )

        if document_stats["error"]:

            st.code(
                document_stats["error"],
                language="text"
            )

    else:

        # ----------------------------------------------------
        # MAIN NUMBERS
        # ----------------------------------------------------

        st.subheader("Document Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            metric_card(
                "Source Text",
                f"{document_stats['text_length']:,}",
                "Characters extracted from PDF"
            )

        with col2:

            metric_card(
                "Total Chunks",
                f"{len(chunk_lengths):,}",
                "Chunks generated"
            )

        with col3:

            metric_card(
                "Minimum Chunk",
                f"{min(chunk_lengths):,}",
                "Characters"
            )

        with col4:

            metric_card(
                "Maximum Chunk",
                f"{max(chunk_lengths):,}",
                "Characters"
            )

        st.write("")

        col1, col2, col3, col4 = st.columns(4)

        average_length = (
            sum(chunk_lengths)
            / len(chunk_lengths)
        )

        median_length = (
            pd.Series(chunk_lengths)
            .median()
        )

        with col1:

            metric_card(
                "Average Chunk",
                f"{average_length:.2f}",
                "Characters"
            )

        with col2:

            metric_card(
                "Median Chunk",
                f"{median_length:.0f}",
                "Characters"
            )

        with col3:

            metric_card(
                "Chunk Size",
                "800",
                "Configured maximum"
            )

        with col4:

            metric_card(
                "Overlap",
                "150",
                "Configured overlap"
            )

        # ----------------------------------------------------
        # CHUNKING FORMULA
        # ----------------------------------------------------

        st.subheader("Chunking Logic")

        st.code(
            """
Chunk Size = 800
Overlap = 150

Step Size = Chunk Size - Overlap
          = 800 - 150
          = 650
""",
            language="text"
        )

        st.write(
            "Every new chunk begins 650 characters after the "
            "previous chunk starts. The final 150 characters of "
            "one chunk can therefore overlap with the beginning "
            "of the next chunk."
        )

        # ----------------------------------------------------
        # CHUNK STATISTICS
        # ----------------------------------------------------

        st.subheader("Chunk Statistics")

        chunk_stats = pd.DataFrame(
            [
                {
                    "Statistic": "Total Chunks",
                    "Value": len(chunk_lengths)
                },
                {
                    "Statistic": "Chunk Size",
                    "Value": 800
                },
                {
                    "Statistic": "Chunk Overlap",
                    "Value": 150
                },
                {
                    "Statistic": "Step Size",
                    "Value": 650
                },
                {
                    "Statistic": "Minimum",
                    "Value": min(chunk_lengths)
                },
                {
                    "Statistic": "Maximum",
                    "Value": max(chunk_lengths)
                },
                {
                    "Statistic": "Average",
                    "Value": round(
                        average_length,
                        2
                    )
                },
                {
                    "Statistic": "Median",
                    "Value": round(
                        median_length,
                        2
                    )
                }
            ]
        )

        st.dataframe(
            chunk_stats,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Chunk Length Distribution")

        ranges = [
            (0, 599),
            (600, 649),
            (650, 699),
            (700, 749),
            (750, 799),
            (800, 800)
        ]

        rows = []

        for lower, upper in ranges:

            count = sum(
                lower <= length <= upper
                for length in chunk_lengths
            )

            percentage = (
                count
                / len(chunk_lengths)
                * 100
            )

            rows.append(
                {
                    "Length Range":
                        f"{lower} - {upper}",
                    "Chunks":
                        count,
                    "Percentage":
                        f"{percentage:.2f}%"
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 4
# EVALUATION & HALLUCINATION
# ============================================================

elif page == "Evaluation & Hallucination":

    page_header(
        "Evaluation & Hallucination",
        "Understand answer quality, faithfulness, context quality, "
        "and the dashboard-derived hallucination diagnostic."
    )

    # --------------------------------------------------------
    # TOP SCORES
    # --------------------------------------------------------

    st.subheader("Overall Evaluation")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card(
            "Faithfulness",
            f"{faithfulness:.4f}",
            "RAGAS metric"
        )

    with col2:

        metric_card(
            "Hallucination Diagnostic",
            f"{hallucination_diagnostic:.4f}",
            "1 - Faithfulness"
        )

    with col3:

        metric_card(
            "Answer Relevancy",
            f"{answer_relevancy:.4f}",
            "RAGAS metric"
        )

    with col4:

        metric_card(
            "Context Precision",
            f"{context_precision:.4f}",
            "RAGAS metric"
        )

    with col5:

        metric_card(
            "Context Recall",
            f"{context_recall:.4f}",
            "RAGAS metric"
        )

    # --------------------------------------------------------
    # HALLUCINATION EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "Hallucination / Unsupported-Claim Diagnostic"
    )

    st.write(
        "Faithfulness measures whether generated claims are "
        "supported by the retrieved context."
    )

    st.write(
        "This dashboard derives an unsupported-claim diagnostic "
        "using:"
    )

    st.code(
        "Unsupported-Claim Diagnostic = 1 - Faithfulness",
        language="text"
    )

    st.write(
        f"For the current evaluation:"
    )

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            "Faithfulness",
            f"{faithfulness:.4f}",
            "Average RAGAS Faithfulness"
        )

    with col2:

        metric_card(
            "Derived Diagnostic",
            f"{hallucination_diagnostic:.4f}",
            "1 - Faithfulness"
        )

    st.caption(
        "Important: this derived value should not be presented "
        "as an official RAGAS hallucination score."
    )

    # --------------------------------------------------------
    # SCORE BARS
    # --------------------------------------------------------

    st.subheader("Metric Scores")

    metrics = [
        ("Faithfulness", faithfulness),
        ("Answer Relevancy", answer_relevancy),
        ("Context Precision", context_precision),
        ("Context Recall", context_recall)
    ]

    for label, score in metrics:

        st.write(
            f"{label}: {score:.4f}"
        )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    score
                )
            )
        )

    # --------------------------------------------------------
    # METRIC DETAILS
    # --------------------------------------------------------

    st.subheader("Metric Details")

    details = []

    for display_name, column_name in [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall")
    ]:

        details.append(
            {
                "Metric": display_name,
                "Average":
                    round(
                        df[column_name].mean(),
                        4
                    ),
                "Minimum":
                    round(
                        df[column_name].min(),
                        4
                    ),
                "Maximum":
                    round(
                        df[column_name].max(),
                        4
                    )
            }
        )

    st.dataframe(
        pd.DataFrame(details),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # QUESTION LEVEL
    # --------------------------------------------------------

    st.subheader("Question-Level Evaluation")

    question_table = df[
        [
            "question",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]
    ].copy()

    question_table.insert(
        0,
        "ID",
        [
            f"Q{i}"
            for i in range(
                1,
                len(question_table) + 1
            )
        ]
    )

    question_table = question_table.rename(
        columns={
            "question": "Question",
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall"
        }
    )

    for column in [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall"
    ]:

        question_table[column] = (
            question_table[column]
            .round(4)
        )

    st.dataframe(
        question_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 5
# ANSWER INSPECTION
# ============================================================

elif page == "Answer Inspection":

    page_header(
        "Answer Inspection",
        "Inspect an individual generated answer, its evaluation "
        "scores, and the contexts used by the evaluator."
    )

    # --------------------------------------------------------
    # QUESTION SELECTOR
    # --------------------------------------------------------

    selected_question = st.selectbox(
        "Select evaluation question",
        df["question"].tolist()
    )

    selected_row = df[
        df["question"] == selected_question
    ].iloc[0]

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    st.subheader("Question")

    st.write(
        selected_question
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.subheader("Generated Answer")

    st.write(
        selected_row["answer"]
    )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    st.subheader("Answer Evaluation")

    selected_faithfulness = float(
        selected_row["faithfulness"]
    )

    selected_relevancy = float(
        selected_row["answer_relevancy"]
    )

    selected_precision = float(
        selected_row["context_precision"]
    )

    selected_recall = float(
        selected_row["context_recall"]
    )

    selected_hallucination = (
        1 - selected_faithfulness
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card(
            "Faithfulness",
            f"{selected_faithfulness:.4f}",
            "RAGAS"
        )

    with col2:

        metric_card(
            "Hallucination Diagnostic",
            f"{selected_hallucination:.4f}",
            "1 - Faithfulness"
        )

    with col3:

        metric_card(
            "Answer Relevancy",
            f"{selected_relevancy:.4f}",
            "RAGAS"
        )

    with col4:

        metric_card(
            "Context Precision",
            f"{selected_precision:.4f}",
            "RAGAS"
        )

    with col5:

        metric_card(
            "Context Recall",
            f"{selected_recall:.4f}",
            "RAGAS"
        )

    st.caption(
        "The hallucination value shown here is a derived "
        "unsupported-claim diagnostic, not an official RAGAS metric."
    )

    # --------------------------------------------------------
    # CONTEXTS
    # --------------------------------------------------------

    st.subheader("Retrieved Contexts")

    contexts = selected_row.get(
        "retrieved_contexts",
        []
    )

    if not contexts:

        st.write(
            "No retrieved contexts are stored in the evaluation CSV."
        )

    else:

        for index, context in enumerate(
            contexts,
            start=1
        ):

            with st.expander(
                f"Context {index}"
            ):

                st.write(
                    context
                )


# ============================================================
# PAGE 6
# RETRIEVAL LAB
# ============================================================

elif page == "Retrieval Lab":

    page_header(
        "Retrieval Lab",
        "Run a query through dense retrieval and cross-encoder "
        "reranking and inspect exactly what changed."
    )

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query = st.text_area(
        "Query",
        value=(
            "What is NVIDIA focused on "
            "in artificial intelligence?"
        ),
        height=100
    )

    run_button = st.button(
        "Run Retrieval Analysis",
        type="primary"
    )

    if run_button:

        if not query.strip():

            st.write(
                "Please enter a query."
            )

        else:

            with st.spinner(
                "Running retrieval pipeline..."
            ):

                pipeline = (
                    load_retrieval_pipeline()
                )

                query_embedding = (
                    pipeline.embedding_model.encode(
                        [query],
                        convert_to_numpy=True
                    )
                )

                dense_results = (
                    pipeline.retriever.search(
                        query_embedding,
                        top_k=8
                    )
                )

                reranked_results = (
                    pipeline.reranker.rerank(
                        query,
                        dense_results,
                        top_k=5
                    )
                )

            # ------------------------------------------------
            # CALCULATIONS
            # ------------------------------------------------

            dense_scores = [
                item["score"]
                for item in dense_results
            ]

            rerank_scores = [
                item["rerank_score"]
                for item in reranked_results
            ]

            top_dense = (
                max(dense_scores)
                if dense_scores
                else 0
            )

            average_dense = (
                sum(dense_scores)
                / len(dense_scores)
                if dense_scores
                else 0
            )

            top_rerank = (
                max(rerank_scores)
                if rerank_scores
                else 0
            )

            promoted = sum(
                item["rank"] > item["rerank_rank"]
                for item in reranked_results
            )

            demoted = sum(
                item["rank"] < item["rerank_rank"]
                for item in reranked_results
            )

            unchanged = sum(
                item["rank"] == item["rerank_rank"]
                for item in reranked_results
            )

            total_movement = sum(
                abs(
                    item["rank"]
                    - item["rerank_rank"]
                )
                for item in reranked_results
            )

            average_movement = (
                total_movement
                / len(reranked_results)
                if reranked_results
                else 0
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            st.session_state.last_retrieval = {
                "query": query,
                "dense_results": dense_results,
                "reranked_results": reranked_results,
                "top_dense": top_dense,
                "average_dense": average_dense,
                "top_rerank": top_rerank,
                "promoted": promoted,
                "demoted": demoted,
                "unchanged": unchanged,
                "total_movement": total_movement,
                "average_movement": average_movement
            }

            history_queries = [
                item["Query"]
                for item in st.session_state.retrieval_history
            ]

            if query not in history_queries:

                st.session_state.retrieval_history.append(
                    {
                        "Query": query,
                        "Dense Top Score":
                            round(
                                top_dense,
                                4
                            ),
                        "Dense Average Score":
                            round(
                                average_dense,
                                4
                            ),
                        "Top Rerank Score":
                            round(
                                top_rerank,
                                4
                            ),
                        "Candidates":
                            len(dense_results),
                        "Final Contexts":
                            len(reranked_results),
                        "Promoted":
                            promoted,
                        "Demoted":
                            demoted,
                        "Unchanged":
                            unchanged,
                        "Total Rank Movement":
                            total_movement,
                        "Average Rank Movement":
                            round(
                                average_movement,
                                2
                            )
                    }
                )

    # --------------------------------------------------------
    # DISPLAY LAST RESULT
    # --------------------------------------------------------

    result = st.session_state.last_retrieval

    if result is None:

        st.write(
            "No retrieval query has been run yet."
        )

    else:

        st.divider()

        st.subheader("Query")

        st.write(
            result["query"]
        )

        # ----------------------------------------------------
        # DENSE RETRIEVAL
        # ----------------------------------------------------

        st.subheader("Stage 1: Dense Retrieval")

        col1, col2, col3 = st.columns(3)

        with col1:

            metric_card(
                "Retrieved",
                str(
                    len(
                        result["dense_results"]
                    )
                ),
                "Candidate documents"
            )

        with col2:

            metric_card(
                "Top Similarity",
                f"{result['top_dense']:.4f}",
                "Cosine similarity"
            )

        with col3:

            metric_card(
                "Average Similarity",
                f"{result['average_dense']:.4f}",
                "Across 8 candidates"
            )

        dense_table = pd.DataFrame(
            [
                {
                    "Dense Rank":
                        item["rank"],
                    "Similarity":
                        round(
                            item["score"],
                            4
                        )
                }
                for item in result["dense_results"]
            ]
        )

        st.dataframe(
            dense_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # RERANKING
        # ----------------------------------------------------

        st.subheader(
            "Stage 2: Cross-Encoder Reranking"
        )

        st.write(
            "The cross-encoder receives the query and each "
            "candidate document and produces a new relevance "
            "score. These scores use a different scale from "
            "dense similarity."
        )

        rerank_table = pd.DataFrame(
            [
                {
                    "Final Rank":
                        item["rerank_rank"],
                    "Original Dense Rank":
                        item["rank"],
                    "Rank Movement":
                        (
                            item["rank"]
                            - item["rerank_rank"]
                        ),
                    "Dense Score":
                        round(
                            item["score"],
                            4
                        ),
                    "Rerank Score":
                        round(
                            item["rerank_score"],
                            4
                        )
                }
                for item in result["reranked_results"]
            ]
        )

        st.dataframe(
            rerank_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # MOVEMENT
        # ----------------------------------------------------

        st.subheader("What Changed After Reranking?")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            metric_card(
                "Promoted",
                str(result["promoted"]),
                "Moved upward"
            )

        with col2:

            metric_card(
                "Demoted",
                str(result["demoted"]),
                "Moved downward"
            )

        with col3:

            metric_card(
                "Unchanged",
                str(result["unchanged"]),
                "Same rank"
            )

        with col4:

            metric_card(
                "Total Movement",
                str(result["total_movement"]),
                "Sum of rank changes"
            )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        st.subheader("Retrieval Diagnostics")

        dense_scores = [
            item["score"]
            for item in result["dense_results"]
        ]

        rerank_scores = [
            item["rerank_score"]
            for item in result["reranked_results"]
        ]

        diagnostic_table = pd.DataFrame(
            [
                {
                    "Diagnostic": "Dense Score Range",
                    "Value":
                        f"{min(dense_scores):.4f} "
                        f"to {max(dense_scores):.4f}"
                },
                {
                    "Diagnostic": "Cross-Encoder Score Range",
                    "Value":
                        f"{min(rerank_scores):.4f} "
                        f"to {max(rerank_scores):.4f}"
                },
                {
                    "Diagnostic": "Candidate Pool",
                    "Value":
                        len(result["dense_results"])
                },
                {
                    "Diagnostic": "Final Context Set",
                    "Value":
                        len(result["reranked_results"])
                },
                {
                    "Diagnostic": "Removed After Reranking",
                    "Value":
                        max(
                            0,
                            len(result["dense_results"])
                            - len(result["reranked_results"])
                        )
                },
                {
                    "Diagnostic": "Average Rank Movement",
                    "Value":
                        f"{result['average_movement']:.2f}"
                }
            ]
        )

        st.dataframe(
            diagnostic_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CONTEXT DETAILS
        # ----------------------------------------------------

        st.subheader("Final Retrieved Contexts")

        for item in result["reranked_results"]:

            movement = (
                item["rank"]
                - item["rerank_rank"]
            )

            if movement > 0:

                status = "Promoted"

            elif movement < 0:

                status = "Demoted"

            else:

                status = "Unchanged"

            with st.expander(
                f"Final Rank {item['rerank_rank']} | "
                f"Dense Rank {item['rank']} | "
                f"{status}"
            ):

                st.write(
                    f"Dense similarity: "
                    f"{item['score']:.4f}"
                )

                st.write(
                    f"Cross-encoder score: "
                    f"{item['rerank_score']:.4f}"
                )

                st.write(
                    f"Rank movement: "
                    f"{movement:+d}"
                )

                st.divider()

                st.write(
                    item["text"]
                )


# ============================================================
# PAGE 7
# QUERY ANALYSIS
# ============================================================

elif page == "Query Analysis":

    page_header(
        "Query Analysis",
        "Compare retrieval behaviour across queries run "
        "during this dashboard session."
    )

    history = (
        st.session_state.retrieval_history
    )

    if not history:

        st.write(
            "No retrieval queries have been recorded yet."
        )

        st.write(
            "Go to Retrieval Lab and run queries first."
        )

    else:

        history_df = pd.DataFrame(
            history
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("Session Summary")

        col1, col2, col3 = st.columns(3)

        with col1:

            metric_card(
                "Queries Run",
                str(len(history_df)),
                "Unique queries"
            )

        with col2:

            metric_card(
                "Average Dense Top Score",
                f"{history_df['Dense Top Score'].mean():.4f}",
                "Across recorded queries"
            )

        with col3:

            metric_card(
                "Average Rank Movement",
                f"{history_df['Average Rank Movement'].mean():.2f}",
                "Across recorded queries"
            )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.subheader("Query History")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SCORE COMPARISON
        # ----------------------------------------------------

        st.subheader("Retrieval Score Comparison")

        score_comparison = history_df[
            [
                "Query",
                "Dense Top Score",
                "Dense Average Score",
                "Top Rerank Score"
            ]
        ]

        st.dataframe(
            score_comparison,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # MOVEMENT COMPARISON
        # ----------------------------------------------------

        st.subheader("Rank Movement Comparison")

        movement_comparison = history_df[
            [
                "Query",
                "Promoted",
                "Demoted",
                "Unchanged",
                "Total Rank Movement",
                "Average Rank Movement"
            ]
        ]

        st.dataframe(
            movement_comparison,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "Clear Query History"
        ):

            st.session_state.retrieval_history = []

            st.session_state.last_retrieval = None

            st.rerun()


