"""
Ticker Query – Streamlit App  (fully standalone)
=================================================
No src/ imports. Loads only plain-data artifacts produced by vectorize_data.py
and packaged by prepare_deploy.py.

Expected repo layout:
  app/query_vectors_app.py
  data/
    {run_name}/
      {prefix}_query_config.json
      {prefix}_*_vocab.json
      {prefix}_*_idf.npy
      {prefix}_features.npz  (or .parquet)
      {prefix}_index.parquet
  requirements.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE.parent / "data" / "interim" / "vectorized"


# ── Artifact loading ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_runs(root: Path) -> list[str]:
    """All {category}/{version} dirs under data/interim/vectorized/ with a query_config."""
    found = []
    for p in sorted(root.glob("*/*_query_config.json")):
        found.append(str(p.parent.relative_to(root)))
    return found


@st.cache_data(show_spinner=False)
def load_config(interim_dir: Path) -> tuple[str, dict]:
    """Return (prefix, config_dict) from the query_config.json."""
    files = list(interim_dir.glob("*_query_config.json"))
    if not files:
        raise FileNotFoundError(f"No *_query_config.json found in {interim_dir}")
    path = files[0]
    prefix = path.stem.replace("_query_config", "")
    return prefix, json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_vectors(interim_dir: Path, prefix: str):
    """Load stored ticker feature vectors + index."""
    npz     = interim_dir / f"{prefix}_features.npz"
    parquet = interim_dir / f"{prefix}_features.parquet"
    index_f = interim_dir / f"{prefix}_index.parquet"

    if npz.exists():
        vectors = sp.load_npz(npz)
        index   = pd.read_parquet(index_f).index
    elif parquet.exists():
        df      = pd.read_parquet(parquet)
        vectors = df.values
        index   = df.index
    else:
        raise FileNotFoundError(f"No feature vectors found in {interim_dir}")
    return vectors, index


@st.cache_resource
def get_sbert_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ── Query transform ───────────────────────────────────────────────────────────

def _transform_tfidf(query_text: str, t: dict, interim_dir: Path) -> np.ndarray:
    """Manual TF-IDF: count → (sublinear) → IDF multiply → L2 norm."""
    vocab = json.loads((interim_dir / t["vocab_file"]).read_text())
    idf   = np.load(interim_dir / t["idf_file"])

    cv = CountVectorizer(
        vocabulary=vocab,
        analyzer=t["analyzer"],
        ngram_range=tuple(t["ngram_range"]),
    )
    # CountVectorizer with a vocabulary= arg works without fitting
    cv.vocabulary_ = vocab

    X = cv.transform([query_text]).astype(np.float64)   # (1, vocab_size) sparse

    if t.get("sublinear_tf"):
        X.data = np.log(X.data) + 1

    X = X.multiply(idf)                                  # broadcast IDF
    X = normalize(X, norm="l2")
    return X.toarray()                                   # (1, vocab_size)


def _transform_count(query_text: str, t: dict, interim_dir: Path) -> np.ndarray:
    vocab = json.loads((interim_dir / t["vocab_file"]).read_text())
    cv = CountVectorizer(
        vocabulary=vocab,
        analyzer=t["analyzer"],
        ngram_range=tuple(t["ngram_range"]),
    )
    cv.vocabulary_ = vocab
    X = cv.transform([query_text]).astype(np.float32)
    return normalize(X, norm="l2").toarray()             # (1, vocab_size)


def _transform_sbert(query_text: str, t: dict) -> np.ndarray:
    model  = get_sbert_model(t["model_name"])
    text   = t.get("prefix", "") + query_text
    emb    = model.encode([text], normalize_embeddings=t.get("normalize", True))
    dtype  = np.float32 if t.get("dtype") == "float32" else np.float64
    return emb.astype(dtype)                             # (1, embedding_dim)


def build_query_vector(query_text: str, config: dict, interim_dir: Path) -> np.ndarray:
    """Concatenate all sub-transformer outputs (with weights) into one query vector."""
    parts = []
    for t in config["transformers"]:
        t_type = t["type"]
        weight = float(t.get("weight", 1.0))

        if t_type == "tfidf":
            part = _transform_tfidf(query_text, t, interim_dir)
        elif t_type == "count":
            part = _transform_count(query_text, t, interim_dir)
        elif t_type == "sbert":
            part = _transform_sbert(query_text, t)
        elif t_type == "zeros":
            part = np.zeros((1, t["n_features"]), dtype=np.float32)
        else:
            continue

        parts.append(part * weight)

    return np.hstack(parts)                              # (1, total_dims)


def find_nearest(query_text: str, interim_dir: Path, top_n: int) -> pd.DataFrame:
    prefix, config  = load_config(interim_dir)
    vectors, index  = load_vectors(interim_dir, prefix)
    query_vec       = build_query_vector(query_text, config, interim_dir)
    distances       = cosine_distances(query_vec, vectors).flatten()
    top_idx         = np.argsort(distances)[:top_n]
    return pd.DataFrame({
        "ticker":   index[top_idx],
        "distance": distances[top_idx].round(4),
    }).reset_index(drop=True)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Ticker Query", page_icon="📈", layout="centered")
st.title("📈 Find Closest Tickers")
st.caption("Enter a company description and find the most similar tickers by cosine distance.")

with st.sidebar:
    st.header("Model")

    if not DATA_ROOT.exists():
        st.error(f"Data directory not found:\n`{DATA_ROOT}`")
        st.stop()

    runs = list_runs(DATA_ROOT)
    if not runs:
        st.warning("No exported query artifacts found.\nRun `prepare_deploy.py` first.")
        st.stop()

    selected = st.selectbox("Vectorized run", runs)
    interim_dir = DATA_ROOT / selected
    top_n = st.slider("Top N results", min_value=1, max_value=50, value=10)

query_text = st.text_area(
    "Company description",
    placeholder="e.g. Semiconductor company focused on AI chips and data center GPUs",
    height=120,
)

if st.button("Search", type="primary", disabled=not query_text.strip()):
    with st.spinner("Computing distances…"):
        try:
            results = find_nearest(query_text.strip(), interim_dir, top_n)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            st.stop()

    st.success(f"Top {top_n} closest tickers:")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker":   st.column_config.TextColumn("Ticker"),
            "distance": st.column_config.NumberColumn("Cosine Distance", format="%.4f"),
        },
    )
    st.bar_chart(results.set_index("ticker")["distance"].sort_values(),
                 y_label="Cosine Distance", x_label="Ticker")
