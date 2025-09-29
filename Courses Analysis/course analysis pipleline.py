"""
Analysis pipeline for prompt engineering course dataset
-----------------------------------------------------

This module implements a reproducible end‑to‑end analysis of university
courses related to prompt engineering and large language models.  It
follows the requirements described in the user specification and
produces a collection of cleaned data sets, topic models, figures and
summary tables.  The pipeline is designed to run offline using only
standard Python libraries together with pandas, numpy, scikit‑learn,
matplotlib and nltk.  No external network access is required once the
script is executed.

Key features:

* Robust CSV loading with fallback to latin1 when UTF‑8 fails.
* Cleaning and normalization of categorical fields such as delivery
  mode, course type, offering program and duration.
* Flexible parsing of assessment descriptors into high level
  categories.
* Comprehensive normalisation of the ``Prompt Engineering Topics
  Covered`` column using a synonym dictionary.  The dictionary maps
  many common synonyms and spelling variants onto a canonical topic
  name.  Additional unknown tokens are preserved to avoid data loss.
* Construction of a per‑course topic set as well as an exploded
  long‑form table with one row per topic occurrence.
* Building TF‑IDF representations from both the normalised topics and
  filtered course description snippets.  Dimensionality reduction is
  performed via TruncatedSVD.
* Clustering of courses using KMeans with automatic model selection
  based on silhouette and Davies–Bouldin scores.  HDBSCAN is not
  available in the current environment, so the implementation falls
  back to KMeans gracefully.
* Derivation of a topic co‑occurrence matrix and simple community
  detection via spectral clustering.  Because networkx is not
  available offline, communities are approximated using clustering on
  the adjacency matrix.
* Mapping of topics and clusters to SWEBOK Knowledge Areas using a
  rule based dictionary.  The mapping rules are saved into
  ``taxonomy.json`` and applied to every course.
* Generation of a suite of figures saved into a ``figs/`` directory.
  Each figure is a separate PNG produced with matplotlib.  Colour
  palettes and styles are kept simple and consistent.
* Production of several CSV outputs, including the cleaned data,
  exploded topics, clustering results and SWEBOK mappings.

The code is organised into small functions that encapsulate specific
tasks.  A deterministic random seed is used throughout to ensure
reproducible results.  The top‑level ``main`` function orchestrates
the entire pipeline.  Running this script from the command line
generates all artefacts in the current working directory.
"""

import csv
import json
import os
import re
import string
import warnings
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.manifold import MDS

# Set a global random seed for reproducibility
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV file from `path` with a fallback encoding.

    Parameters
    ----------
    path: str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded data frame.
    """
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are entirely empty (all NaN or blank strings).

    Parameters
    ----------
    df: DataFrame
        Input data frame.

    Returns
    -------
    DataFrame
        Data frame with empty columns removed.
    """
    cols_to_drop = []
    for col in df.columns:
        # Consider a column empty if all values are NaN or empty string
        if df[col].dropna().replace("", np.nan).isna().all():
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


def trim_normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalise unicode punctuation in all string columns.

    This function applies a set of normalisations to all object
    (string) columns: leading/trailing whitespace is stripped, fancy
    quotes and long dashes are replaced with their simple ASCII
    counterparts and multiple spaces are collapsed.  Non‑string
    columns are untouched.

    Parameters
    ----------
    df: DataFrame
        Input data frame.

    Returns
    -------
    DataFrame
        Normalised data frame.
    """
    def normalise_text(val: str) -> str:
        if not isinstance(val, str):
            return val
        # Replace fancy quotes and dashes
        val = val.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes
        val = val.replace("\u2018", "'").replace("\u2019", "'")  # single quotes
        val = val.replace("\u201c", '"').replace("\u201d", '"')  # double quotes
        val = val.replace("\xa0", " ")  # non breaking space
        # Collapse multiple spaces
        val = re.sub(r"\s+", " ", val)
        return val.strip()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(normalise_text)
    return df


def deduplicate_courses(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows based on a combination of Course Name, Offering Institution and Course URL.

    Parameters
    ----------
    df: DataFrame
        Input data frame.

    Returns
    -------
    DataFrame
        Deduplicated data frame.
    """
    subset_cols = [c for c in ["Course Name", "Offering Institution", "Course URL LINK"] if c in df.columns]
    return df.drop_duplicates(subset=subset_cols)


def map_delivery_mode(mode: Optional[str]) -> str:
    """Map a raw delivery mode string to a controlled vocabulary.

    Parameters
    ----------
    mode: str or None
        Raw delivery mode description.

    Returns
    -------
    str
        Normalised delivery mode.
    """
    if not isinstance(mode, str) or not mode:
        return "Unspecified"
    text = mode.lower()
    # Face‑to‑face patterns
    if any(k in text for k in ["face", "campus", "classroom", "on-campus", "f2f"]):
        if "online" in text:
            return "Hybrid"
        return "Face-to-Face"
    # Hybrid patterns
    if any(k in text for k in ["hybrid", "blended", "face-to-face &"]):
        return "Hybrid"
    # Synchronous vs asynchronous online
    if "online" in text:
        if any(k in text for k in ["asynchronous", "on-demand", "self-paced", "on demand"]):
            return "Online (Asynchronous)"
        else:
            return "Online (Synchronous)"
    return "Unspecified"


def map_course_type(raw_type: Optional[str]) -> str:
    """Map raw course type to a controlled set of categories.

    Parameters
    ----------
    raw_type: str or None
        Raw course type description.

    Returns
    -------
    str
        Normalised course type.
    """
    if not isinstance(raw_type, str) or not raw_type:
        return "Other"
    text = raw_type.lower()
    # MOOC / online
    if "mooc" in text or "online course" in text:
        return "MOOC"
    # Degree‑awarding
    if any(k in text for k in ["degree", "undergraduate", "graduate", "course number"]):
        return "Degree-Awarding"
    # Professional certificate
    if "professional certificate" in text or "certificate program" in text:
        return "Professional Certificate"
    # Executive program
    if "executive" in text:
        return "Executive Program"
    # Professional development / short course
    if any(k in text for k in ["short", "workshop", "professional development", "training", "seminar"]):
        if "workshop" in text:
            return "Workshop"
        elif "short" in text:
            return "Short Course"
        else:
            return "Professional Development"
    # Default to Other
    return "Other"


def map_offering_program(raw_prog: Optional[str]) -> str:
    """Map raw offering program to a controlled set.

    Parameters
    ----------
    raw_prog: str or None
        Raw programme description.

    Returns
    -------
    str
        Normalised program category.
    """
    if not isinstance(raw_prog, str) or not raw_prog:
        return "Other"
    text = raw_prog.lower()
    if "undergraduate" in text:
        return "Undergraduate"
    if "graduate" in text:
        return "Graduate"
    if "executive" in text:
        return "Executive Education"
    if "certificate" in text:
        return "Certificate"
    if any(k in text for k in ["short", "workshop"]):
        return "Short Course"
    if "professional" in text and "development" in text:
        return "Professional Development"
    return "Other"


def parse_duration(raw_duration: Optional[str]) -> Tuple[str, Optional[float], Optional[str]]:
    """Parse duration strings into a numeric value and unit.

    Many courses specify durations such as "6 weeks", "8 weeks", "2 days", "12 hours".
    This function attempts to extract the first number and its unit.  If no
    meaningful number is present, the value and unit are returned as None.

    Parameters
    ----------
    raw_duration: str or None
        Raw duration string.

    Returns
    -------
    Tuple[str, float or None, str or None]
        A tuple of (cleaned duration string, numeric value, unit).  The
        cleaned string is trimmed; numeric and unit values may be None.
    """
    if not isinstance(raw_duration, str) or not raw_duration.strip():
        return "Unspecified", None, None
    s = raw_duration.strip()
    # Extract number and unit using regex
    match = re.search(r"([\d.]+)\s*(weeks?|days?|hours?|hrs?|hr|semester|semesters)", s.lower())
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        # Normalise unit
        if unit.startswith("week"):
            unit_norm = "weeks"
        elif unit.startswith("day"):
            unit_norm = "days"
        elif unit.startswith("hour") or unit.startswith("hr"):
            unit_norm = "hours"
        elif unit.startswith("sem"):
            unit_norm = "semesters"
        else:
            unit_norm = unit
        return s, number, unit_norm
    return s, None, None


def parse_assessment(raw_assessment: Optional[str]) -> List[str]:
    """Parse the assessment description into one or more high level categories.

    Recognised categories: Exam, Quiz, Project, Lab/Hands-on, Case Study,
    Participation, Essay/Report, Certificate Only, None/Unspecified.

    Parameters
    ----------
    raw_assessment: str or None
        Raw assessment description.

    Returns
    -------
    List[str]
        List of assessment categories detected.
    """
    if not isinstance(raw_assessment, str) or not raw_assessment.strip():
        return ["None/Unspecified"]
    text = raw_assessment.lower()
    categories = []
    if any(k in text for k in ["exam", "final", "midterm"]):
        categories.append("Exam")
    if "quiz" in text:
        categories.append("Quiz")
    if any(k in text for k in ["project", "projects"]):
        categories.append("Project")
    if any(k in text for k in ["lab", "hands-on", "assignment"]):
        categories.append("Lab/Hands-on")
    if "case" in text:
        categories.append("Case Study")
    if any(k in text for k in ["participation", "attendance"]):
        categories.append("Participation")
    if any(k in text for k in ["essay", "report", "paper"]):
        categories.append("Essay/Report")
    if any(k in text for k in ["certificate"]):
        categories.append("Certificate Only")
    if not categories:
        categories.append("None/Unspecified")
    return list(dict.fromkeys(categories))


def build_synonym_dictionary() -> Dict[str, List[str]]:
    """Define the synonym dictionary for normalising prompt engineering topics.

    The returned dictionary maps each canonical topic to a list of known
    variants and synonyms.  These mappings originate from the user
    specification but can be easily extended here.

    Returns
    -------
    dict
        A mapping from canonical topic names to lists of synonyms.
    """
    synonyms = {
        "chain-of-thought": [
            "chain-of-thought",
            "chain of thought",
            "cot",
            "reasoning traces",
            "scratchpad",
            "deliberate",
            "chain-of-thought reasoning",
            "reasoning",
        ],
        "few-shot": [
            "few shot",
            "few-shot",
            "few-shot learning",
            "in-context examples",
            "few-shot prompts",
        ],
        "zero-shot": [
            "zero shot",
            "zero-shot",
            "zero-shot prompts",
        ],
        "in-context learning": [
            "in-context learning",
            "icl",
            "in context learning",
            "in context",
        ],
        "retrieval-augmented generation": [
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "rag",
            "retrieval augmented generation (rag)",
        ],
        "agentic/agents": [
            "agentic ai",
            "agent-based prompting",
            "tool use",
            "function calling",
            "toolformer",
            "workflow/agents",
            "agents",
            "agentic prompting",
            "agentic ai workflows",
        ],
        "prompt tuning": [
            "prompt-tuning",
            "prompt tuning",
            "p-tuning",
            "prefix tuning",
            "soft prompts",
            "parameter-efficient fine-tuning",
        ],
        "prompt design/fundamentals": [
            "prompt engineering fundamentals",
            "prompt engineering techniques",
            "prompt design",
            "role-based prompts",
            "meta prompts",
            "context-sensitive prompts",
            "iterative prompt refinement",
            "prompt optimization",
            "prompt fundamentals",
            "prompt engineering principles",
            "high-level prompt design",
        ],
        "evaluation/debugging": [
            "eval frameworks",
            "rubrics",
            "metrics",
            "prompt debugging",
            "human-in-the-loop evaluation",
            "evaluation",
            "evaluation metrics",
            "evaluation frameworks",
        ],
        "safety/risk": [
            "hallucination mitigation",
            "guardrails",
            "jailbreak mitigation",
            "toxicity",
            "bias/fairness",
            "red teaming",
            "constitutional ai",
            "safety",
            "ethics",
            "risk",
        ],
        "multimodal prompting": [
            "image prompts",
            "text-to-image",
            "audio prompting",
            "video prompting",
            "multimodal",
            "multimodal prompting",
        ],
        "data techniques": [
            "data augmentation",
            "context window management",
            "retrieval evaluation",
            "data augmentation techniques",
            "context management",
            "data techniques",
        ],
        "applications": [
            "llm applications",
            "business applications",
            "domain-specific prompting",
            "agentic ai workflows",
            "productivity enhancement prompts",
            "workflow optimization",
            "collaborative agents",
            "agents workflows",
        ],
    }
    return synonyms


def invert_synonym_map(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert the synonym dictionary to map each synonym to its canonical topic.

    The canonical keys themselves also map to themselves.  The mapping is
    case‑insensitive.

    Parameters
    ----------
    synonyms: dict
        Mapping from canonical topic names to lists of synonyms.

    Returns
    -------
    dict
        Mapping from synonym to canonical topic.
    """
    mapping: Dict[str, str] = {}
    for canonical, syns in synonyms.items():
        mapping[canonical.lower()] = canonical
        for s in syns:
            mapping[s.lower()] = canonical
    return mapping


def normalise_topic_list(topic_str: Optional[str], inv_map: Dict[str, str]) -> List[str]:
    """Normalise a semi‑structured list of topics into canonical tokens.

    Topics may be separated by semicolons, commas, slashes or pipes.  This
    function tokenises the input, lowercases each token, removes
    surrounding punctuation and maps recognised synonyms onto their
    canonical form.  Unknown tokens are cleaned and included as they
    appear.

    Parameters
    ----------
    topic_str: str or None
        Raw string of topics.
    inv_map: dict
        Inverted synonym map mapping from synonyms to canonical names.

    Returns
    -------
    List[str]
        List of unique canonical topic names for the course.
    """
    if not isinstance(topic_str, str) or not topic_str.strip():
        return []
    # Split on common delimiters
    parts = re.split(r"[;,/|]", topic_str)
    normalised = []
    for part in parts:
        tok = part.strip().lower()
        if not tok:
            continue
        # Remove surrounding quotes/punctuation
        tok = tok.strip(string.punctuation + " ")
        if not tok:
            continue
        # Map synonyms
        canonical = inv_map.get(tok, None)
        if canonical is None:
            # Remove internal extra spaces
            canonical = re.sub(r"\s+", " ", tok)
        normalised.append(canonical)
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for t in normalised:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def clean_normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """Perform all cleaning and normalisation steps on the raw data.

    Parameters
    ----------
    df: DataFrame
        Raw course data.

    Returns
    -------
    Tuple of
        df_clean: DataFrame
            Cleaned and normalised data frame.
        synonyms: dict
            Canonical topic to synonyms mapping.
        inv_map: dict
            Inverted synonym mapping for fast lookup.
    """
    df = drop_empty_columns(df.copy())
    df = trim_normalize_strings(df)
    df = deduplicate_courses(df)
    # Build synonym dictionaries
    synonyms = build_synonym_dictionary()
    inv_map = invert_synonym_map(synonyms)
    # Apply normalisation per row
    # Prepare new columns for cleaned values
    clean_records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        # Delivery mode
        dm_col = "Delivery Mode" if "Delivery Mode" in df.columns else None
        record["Delivery Mode Clean"] = map_delivery_mode(record.get(dm_col)) if dm_col else "Unspecified"
        # Course type
        ct_col = "Course Type" if "Course Type" in df.columns else None
        record["Course Type Clean"] = map_course_type(record.get(ct_col)) if ct_col else "Other"
        # Offering program
        op_col = "Offering Program" if "Offering Program" in df.columns else None
        record["Offering Program Clean"] = map_offering_program(record.get(op_col)) if op_col else "Other"
        # Duration
        dur_col = "Duration" if "Duration" in df.columns else None
        dur_raw = record.get(dur_col) if dur_col else None
        cleaned_dur, dur_num, dur_unit = parse_duration(dur_raw)
        record["Duration Clean"] = cleaned_dur
        record["Duration Value"] = dur_num
        record["Duration Unit"] = dur_unit
        # Assessment
        ass_col = "Assessment" if "Assessment" in df.columns else None
        record["Assessment Categories"] = parse_assessment(record.get(ass_col)) if ass_col else ["None/Unspecified"]
        # Topics
        topics_col = "Prompt Engineering Topics Covered" if "Prompt Engineering Topics Covered" in df.columns else None
        topics_raw = record.get(topics_col) if topics_col else None
        record["Normalized Topics"] = normalise_topic_list(topics_raw, inv_map)
        clean_records.append(record)
    df_clean = pd.DataFrame(clean_records)
    return df_clean, synonyms, inv_map


def explode_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the normalised topic list into a long‑form table.

    Parameters
    ----------
    df: DataFrame
        Cleaned data frame containing a column 'Normalized Topics'.

    Returns
    -------
    DataFrame
        Long form table with each row representing a (course_id, topic) pair.
    """
    records = []
    for idx, row in df.iterrows():
        topics = row.get("Normalized Topics", [])
        for t in topics:
            records.append({"Course Index": idx, "Topic": t})
    return pd.DataFrame(records)


def build_topic_matrix(df: pd.DataFrame, topics_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Create a TF‑IDF matrix from normalised topics.

    Each course's topics are joined into a single string separated by
    spaces.  The vocabulary is the list of all unique normalised
    topics (provided by the caller).  Use scikit‑learn's TfidfVectorizer
    with binary weights (sublinear_tf=False, use_idf=True) to encode
    presence information.

    Parameters
    ----------
    df: DataFrame
        Cleaned data frame with 'Normalized Topics' column.
    topics_list: List[str]
        Sorted list of unique topics to use as vocabulary.

    Returns
    -------
    Tuple of
        X: ndarray
            TF‑IDF matrix (n_courses x n_topics)
        feature_names: List[str]
            Vocabulary used by the vectorizer.
    """
    # Join topics into a single space separated string per course
    corpus = [" ".join(topics) for topics in df["Normalized Topics"]]
    vectorizer = TfidfVectorizer(vocabulary=topics_list, tokenizer=str.split, norm='l2', use_idf=True, sublinear_tf=False)
    X = vectorizer.fit_transform(corpus).toarray()
    return X, vectorizer.get_feature_names_out().tolist()


def build_description_matrix(df: pd.DataFrame, inv_map: Dict[str, str], n_components: int = 100) -> Tuple[np.ndarray, List[str]]:
    """Build TF‑IDF matrix from course descriptions, focusing on sentences containing topics.

    Parameters
    ----------
    df: DataFrame
        Cleaned data frame with 'COURSE DESCRIPTION' and 'Normalized Topics'.
    inv_map: dict
        Inverted synonym mapping for topic detection.
    n_components: int
        Number of components to return in the output (used later for dimensionality reduction).

    Returns
    -------
    Tuple of
        X_desc: ndarray
            TF‑IDF matrix of descriptions (n_courses x n_features).
        feature_names: List[str]
            Feature names (n‑grams) used.
    """
    descriptions = []
    for _, row in df.iterrows():
        desc = row.get("COURSE DESCRIPTION", "") or ""
        desc = str(desc)
        # Split into sentences by period, semicolon or newline
        sentences = re.split(r"[\n\.\;]", desc)
        selected_sentences = []
        # If the course has at least one normalised topic, retain only sentences
        # that contain any synonym token to focus on topical content.
        topics = row.get("Normalized Topics", [])
        # Build a set of tokens for quick match
        tokens_to_match = set(inv_map.keys()) if topics else set()
        if topics:
            for sent in sentences:
                s_lower = sent.lower()
                if any(syn in s_lower for syn in tokens_to_match):
                    selected_sentences.append(sent.strip())
        if not selected_sentences:
            # fallback: use full description if no sentences matched
            selected_sentences = [desc.strip()]
        descriptions.append(". ".join([s for s in selected_sentences if s]))
    # Vectorise using 1–3 gram TF‑IDF
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), min_df=1)
    X_desc = vectorizer.fit_transform(descriptions).toarray()
    return X_desc, vectorizer.get_feature_names_out().tolist()


def reduce_dimensions(X: np.ndarray, n_components: int) -> np.ndarray:
    """Apply TruncatedSVD to reduce dimensionality of the feature matrix.

    Parameters
    ----------
    X: ndarray
        Input matrix.
    n_components: int
        Number of dimensions to retain.

    Returns
    -------
    ndarray
        Reduced representation of shape (n_samples, n_components).
    """
    n_components = min(n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    return svd.fit_transform(X)


def choose_kmeans_clusters(X: np.ndarray, k_range: Iterable[int]) -> Tuple[np.ndarray, Dict[str, float], int]:
    """Fit KMeans models across a range of cluster sizes and choose the best k.

    Uses silhouette score and Davies–Bouldin score to evaluate each
    candidate.  Selects the k that maximises silhouette while
    minimising Davies–Bouldin.  If multiple ks tie, the smallest k
    among the top silhouette scores is chosen to encourage simpler
    solutions.

    Parameters
    ----------
    X: ndarray
        Data to cluster.
    k_range: Iterable[int]
        Range of k values to consider.

    Returns
    -------
    Tuple of
        labels: ndarray
            Cluster labels for the chosen model.
        diagnostics: dict
            Dictionary containing silhouette and Davies–Bouldin scores per k.
        best_k: int
            Selected number of clusters.
    """
    diagnostics = {}
    best_k = None
    best_score = -np.inf
    best_labels = None
    # Compute scores for each k
    for k in k_range:
        if k < 2 or k >= X.shape[0]:
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        # In case of single cluster or empty cluster, skip
        if len(set(labels)) < 2:
            continue
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = float("nan")
        try:
            db = davies_bouldin_score(X, labels)
        except Exception:
            db = float("nan")
        diagnostics[k] = {"silhouette": sil, "davies_bouldin": db}
        # Combined score: high silhouette (positive), low DB (negative)
        combined = sil - db
        if combined > best_score:
            best_score = combined
            best_k = k
            best_labels = labels
    # If no suitable k found (e.g., all clusters invalid), fall back to 1 cluster
    if best_labels is None:
        best_k = 1
        best_labels = np.zeros(X.shape[0], dtype=int)
        diagnostics[1] = {"silhouette": float("nan"), "davies_bouldin": float("nan")}
    return best_labels, diagnostics, best_k


def cluster_courses(X_topics: np.ndarray, X_desc: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], int, np.ndarray, Dict[str, float], int]:
    """Cluster courses based on topics and descriptions separately.

    Parameters
    ----------
    X_topics: ndarray
        TF‑IDF matrix of topics.
    X_desc: ndarray
        TF‑IDF matrix of filtered descriptions.

    Returns
    -------
    Tuple of
        labels_topics: ndarray
            Cluster labels for topic representation.
        diagnostics_topics: dict
            Diagnostics for topic clustering.
        best_k_topics: int
            Selected cluster count for topics.
        labels_desc: ndarray
            Cluster labels for description representation.
        diagnostics_desc: dict
            Diagnostics for description clustering.
        best_k_desc: int
            Selected cluster count for descriptions.
    """
    # Reduce dimensions before clustering to 50 for topics and 100 for descriptions
    X_topics_red = reduce_dimensions(X_topics, n_components=min(50, X_topics.shape[1]-1) if X_topics.shape[1] > 1 else 1)
    X_desc_red = reduce_dimensions(X_desc, n_components=min(100, X_desc.shape[1]-1) if X_desc.shape[1] > 1 else 1)
    # Determine k range; ensure at least 2 and at most half of samples
    n_samples = X_topics_red.shape[0]
    max_k = min(12, max(2, n_samples // 2))
    k_range = range(2, max_k + 1)
    labels_topics, diagnostics_topics, best_k_topics = choose_kmeans_clusters(X_topics_red, k_range)
    labels_desc, diagnostics_desc, best_k_desc = choose_kmeans_clusters(X_desc_red, k_range)
    return labels_topics, diagnostics_topics, best_k_topics, labels_desc, diagnostics_desc, best_k_desc


def compute_cooccurrence(df_topics: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Compute topic co‑occurrence matrix.

    Parameters
    ----------
    df_topics: DataFrame
        Long form table of course index and topics.

    Returns
    -------
    Tuple of
        co_matrix: ndarray
            Symmetric co‑occurrence counts matrix.
        topic_list: List[str]
            List of unique topics in order corresponding to the matrix.
    """
    topics = df_topics['Topic'].unique().tolist()
    idx_map = {t: i for i, t in enumerate(topics)}
    n = len(topics)
    co_matrix = np.zeros((n, n), dtype=int)
    # Build co‑occurrence by grouping by course
    grouped = df_topics.groupby('Course Index')['Topic'].apply(list)
    for topic_list in grouped:
        # Unique topics in course to avoid double counting within same course
        unique_topics = list(dict.fromkeys(topic_list))
        for i in range(len(unique_topics)):
            for j in range(i + 1, len(unique_topics)):
                a = idx_map[unique_topics[i]]
                b = idx_map[unique_topics[j]]
                co_matrix[a, b] += 1
                co_matrix[b, a] += 1
    return co_matrix, topics


def detect_communities(co_matrix: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Approximate community detection using spectral clustering.

    Because networkx is unavailable, we approximate community structure
    by performing spectral clustering on the co‑occurrence adjacency matrix.
    The number of communities is chosen heuristically (default 4).  If
    there are fewer than 4 topics, we assign all topics to a single
    community.

    Parameters
    ----------
    co_matrix: ndarray
        Co‑occurrence matrix.
    n_clusters: int
        Number of communities to detect.

    Returns
    -------
    ndarray
        Community label per topic.
    """
    n_topics = co_matrix.shape[0]
    if n_topics < 2:
        return np.zeros(n_topics, dtype=int)
    # Normalise matrix to build similarity (laplacian may need positive
    # values).  Add a small constant to avoid zeros.
    sim = co_matrix + np.eye(n_topics)
    # Determine number of clusters no larger than number of topics
    n_clusters = min(n_clusters, n_topics)
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=RANDOM_STATE)
        labels = clustering.fit_predict(sim)
    except Exception:
        # Fallback: all in one community
        labels = np.zeros(n_topics, dtype=int)
    return labels


def compute_swebok_mapping(topics: Iterable[str]) -> Dict[str, List[str]]:
    """Return a rule‑based mapping from topics to SWEBOK knowledge areas.

    The mapping is based on the specification provided by the user.  Each
    topic may map to multiple knowledge areas.  Unknown topics map to
    an empty list.

    Parameters
    ----------
    topics: Iterable[str]
        Collection of unique canonical topics.

    Returns
    -------
    dict
        Mapping from topic to list of SWEBOK KAs.
    """
    mapping = {}
    for t in topics:
        cats = []
        tl = t.lower()
        # Requirements: elicitation/context needs, acceptance criteria
        if any(k in tl for k in ["requirements", "context", "needs", "acceptance"]):
            cats.append("Requirements")
        # Design: architecture, roles, workflow, integration
        if any(k in tl for k in ["design", "role", "workflow", "agent", "rag", "retrieval", "tool"]):
            cats.append("Design")
        # Construction: writing, structuring prompts, tuning
        if any(k in tl for k in ["prompt", "tuning", "optimization", "p-tuning", "prefix"]):
            cats.append("Construction")
        # Testing/Evaluation
        if any(k in tl for k in ["evaluation", "debug", "metric", "rubric", "test"]):
            cats.append("Testing")
        # Maintenance: lifecycle, versioning, monitoring
        if any(k in tl for k in ["lifecycle", "version", "monitor", "maintenance"]):
            cats.append("Maintenance")
        # Configuration management
        if any(k in tl for k in ["repository", "dataset", "configuration", "context window"]):
            cats.append("Configuration Management")
        # Engineering management: risk, scheduling, governance
        if any(k in tl for k in ["risk", "governance", "management", "schedule"]):
            cats.append("Engineering Management")
        # Process: pipelines, llmops
        if any(k in tl for k in ["pipeline", "process", "ops", "llmops", "workflow"]):
            cats.append("Process")
        # Models & Methods
        if any(k in tl for k in ["chain-of-thought", "cot", "self-consistency", "few-shot", "zero-shot", "in-context"]):
            cats.append("Models & Methods")
        # Quality
        if any(k in tl for k in ["quality", "robust", "bias", "hallucination", "guardrails", "fairness", "safety"]):
            cats.append("Quality")
        # Professional Practice / Ethics
        if any(k in tl for k in ["ethic", "bias", "fairness", "safety", "risk"]):
            cats.append("Professional Practice/Ethics")
        # Economics
        if any(k in tl for k in ["cost", "efficiency", "token", "latency"]):
            cats.append("Economics")
        # Foundations
        if any(k in tl for k in ["transformer", "llm", "architecture", "fine-tuning", "rlhf", "foundation"]):
            cats.append("Foundations")
        # Remove duplicates
        mapping[t] = list(dict.fromkeys(cats))
    return mapping


def assign_swebok(df: pd.DataFrame, topic_to_swebok: Dict[str, List[str]]) -> pd.DataFrame:
    """Assign SWEBOK KAs to each course based on its topics.

    For each course the list of knowledge areas is the union of the
    knowledge areas associated with each of its topics.  A score is
    computed for each KA equal to the proportion of the course's topics
    that map to that KA.  Courses with no mapped topics receive
    empty lists.

    Parameters
    ----------
    df: DataFrame
        Cleaned data frame with 'Normalized Topics'.
    topic_to_swebok: dict
        Mapping from canonical topic to SWEBOK KAs.

    Returns
    -------
    DataFrame
        Data frame with additional columns 'SWEBOK Areas' and
        'SWEBOK Scores' (dict of KA to score).
    """
    swebok_areas = []
    swebok_scores = []
    for _, row in df.iterrows():
        topics = row.get("Normalized Topics", [])
        areas_count: Counter = Counter()
        total_topics = len(topics)
        for t in topics:
            kas = topic_to_swebok.get(t, [])
            for ka in kas:
                areas_count[ka] += 1
        # Compute scores
        if total_topics > 0:
            scores = {ka: count / total_topics for ka, count in areas_count.items()}
        else:
            scores = {}
        swebok_areas.append(list(areas_count.keys()))
        swebok_scores.append(scores)
    df = df.copy()
    df["SWEBOK Areas"] = swebok_areas
    df["SWEBOK Scores"] = swebok_scores
    return df


def save_taxonomy_json(vocab: List[str], synonyms: Dict[str, List[str]], theme_mapping: Dict[str, str], swebok_map: Dict[str, List[str]], path: str) -> None:
    """Save the taxonomy information to a JSON file.

    The JSON file contains the list of canonical topics, a mapping from
    canonical topics to their synonyms, a mapping of canonical topics
    to high level themes, and a mapping to SWEBOK KAs.

    Parameters
    ----------
    vocab: List[str]
        Sorted list of canonical topics.
    synonyms: dict
        Canonical topic to synonyms mapping.
    theme_mapping: dict
        Canonical topic to high level theme name mapping.
    swebok_map: dict
        Canonical topic to list of SWEBOK KAs mapping.
    path: str
        File path to write JSON.
    """
    taxonomy = {
        "vocabulary": vocab,
        "synonyms": synonyms,
        "themes": theme_mapping,
        "swebok_mapping": swebok_map,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)


def assign_themes(topics: Iterable[str]) -> Dict[str, str]:
    """Assign a high level theme to each canonical topic.

    Themes are defined according to the user specification.  Topics
    falling outside defined themes are assigned to 'Other'.

    Parameters
    ----------
    topics: iterable of str
        Canonical topics.

    Returns
    -------
    dict
        Mapping from canonical topic to theme name.
    """
    # Define theme keywords sets
    themes_def = {
        "Prompt Fundamentals & Design": ["prompt design/fundamentals", "prompt engineering principles", "high-level prompt design"],
        "Reasoning & Chain-of-Thought": ["chain-of-thought", "chain-of-thought reasoning"],
        "In-Context Learning (Zero/Few-shot)": ["few-shot", "zero-shot", "in-context learning"],
        "Retrieval & Tool/Function Use": ["retrieval-augmented generation", "agentic/agents", "function calling", "tool use"],
        "Optimization & Tuning": ["prompt tuning", "p-tuning", "prefix tuning"],
        "Evaluation & Debugging": ["evaluation/debugging", "metrics", "rubrics", "debugging"],
        "Safety, Risk & Governance": ["safety/risk", "bias", "fairness", "guardrails", "hallucination"],
        "Multimodal Prompting": ["multimodal prompting", "text-to-image", "audio prompting", "video prompting"],
        "Data & Context Management": ["data techniques", "data augmentation", "context window"],
        "Applications & Domain-specific": ["applications", "llm applications", "business applications", "domain-specific"]
    }
    theme_mapping = {}
    for topic in topics:
        assigned = "Other"
        for theme, keys in themes_def.items():
            if any(key in topic.lower() for key in keys):
                assigned = theme
                break
        theme_mapping[topic] = assigned
    return theme_mapping


def make_figures(df: pd.DataFrame, df_topics: pd.DataFrame, co_matrix: np.ndarray, co_topics: List[str], topic_comm: np.ndarray, labels_topics: np.ndarray, labels_desc: np.ndarray, swebok_map: Dict[str, List[str]], themes: Dict[str, str], outdir: str) -> None:
    """Generate required figures and save them to the specified directory.

    Parameters
    ----------
    df: DataFrame
        Cleaned data frame with derived columns.
    df_topics: DataFrame
        Exploded topics table.
    co_matrix: ndarray
        Topic co‑occurrence matrix.
    co_topics: List[str]
        List of topics corresponding to the co‑occurrence matrix.
    topic_comm: ndarray
        Community labels for topics.
    labels_topics: ndarray
        Cluster labels from the topics representation.
    labels_desc: ndarray
        Cluster labels from the descriptions representation.
    swebok_map: Dict[str, List[str]]
        Mapping from topic to SWEBOK KAs.
    themes: Dict[str, str]
        Mapping from topic to high level theme.
    outdir: str
        Directory to save the figures.
    """
    os.makedirs(outdir, exist_ok=True)
    # Plot missingness per column
    missing = df.isna().sum() + (df == "").sum()
    plt.figure(figsize=(8, 4))
    missing.sort_values(ascending=False).plot(kind='bar')
    plt.ylabel("Number of missing values")
    plt.title("Missingness by column")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "missingness.png"))
    plt.close()
    # Counts by Course Type, Offering Program, Delivery Mode
    for col, title, fname in [
        ("Course Type Clean", "Course Count by Type", "counts_course_type.png"),
        ("Offering Program Clean", "Course Count by Program", "counts_program.png"),
        ("Delivery Mode Clean", "Course Count by Delivery Mode", "counts_delivery_mode.png"),
    ]:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.ylabel("Number of courses")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    # Top topics bar chart
    topic_counts = df_topics['Topic'].value_counts().sort_values(ascending=False)
    top_n = topic_counts.head(20)
    plt.figure(figsize=(10, 5))
    top_n.plot(kind='bar')
    plt.ylabel("Frequency")
    plt.title("Top 20 Normalised Topics")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_topics.png"))
    plt.close()
    # 2D SVD scatter of courses coloured by topic clusters
    # Derive 2D projection from topic TF‑IDF matrix used earlier
    # For reproducibility, recompute projection using X_topics
    all_topics = sorted(df_topics['Topic'].unique())
    X_topics_full, _ = build_topic_matrix(df, all_topics)
    X_svd = reduce_dimensions(X_topics_full, n_components=2)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=labels_topics, cmap='tab10', s=40, alpha=0.8)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("2D Projection of Courses by Topic Clusters")
    # Create legend with unique cluster labels
    unique_labels = np.unique(labels_topics)
    handles = []
    for lab in unique_labels:
        handles.append(plt.Line2D([], [], marker='o', linestyle='', color=plt.cm.tab10(lab / max(unique_labels.max(), 1)), label=f"Cluster {lab}"))
    plt.legend(handles=handles, title="Topic Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "topic_clusters_scatter.png"))
    plt.close()
    # Co‑occurrence network plot using MDS for layout
    if co_matrix.shape[0] > 1:
        # Compute 2D positions using classical multidimensional scaling
        # Convert co‑occurrence counts to distances (inverse) with added epsilon
        max_co = np.max(co_matrix) + 1e-6
        dist_matrix = max_co - co_matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=RANDOM_STATE)
        coords = mds.fit_transform(dist_matrix)
        degrees = co_matrix.sum(axis=0)
        # Determine edges above threshold for visualisation
        threshold = np.percentile(co_matrix[co_matrix > 0], 75) if (co_matrix > 0).any() else 0
        plt.figure(figsize=(10, 8))
        # Draw edges
        for i in range(len(co_topics)):
            for j in range(i + 1, len(co_topics)):
                weight = co_matrix[i, j]
                if weight > threshold:
                    plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                             linewidth=0.5 + 2 * (weight / max_co), color='lightgray', alpha=0.7)
        # Draw nodes
        sizes = 50 + 200 * (degrees / (degrees.max() + 1e-6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], s=sizes, c=topic_comm, cmap='tab10', alpha=0.8)
        # Annotate top hubs (top 10 degrees)
        top_indices = np.argsort(-degrees)[:10]
        for idx in top_indices:
            plt.text(coords[idx, 0], coords[idx, 1], co_topics[idx], fontsize=8, ha='center', va='center')
        plt.title("Topic Co-occurrence Network (Hubs annotated)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "cooccurrence_network.png"))
        plt.close()
    # Heatmap: Themes × Institutions
    # Pivot table counts of themes by institution
    theme_series = df['Normalized Topics'].apply(lambda ts: [themes.get(t, 'Other') for t in ts])
    # Flatten to list of (institution, theme)
    rows = []
    for inst, tlist in zip(df['Offering Institution'], theme_series):
        for t in set(tlist):
            rows.append({"Institution": inst, "Theme": t})
    if rows:
        heat_df = pd.DataFrame(rows)
        heat_table = pd.crosstab(heat_df['Theme'], heat_df['Institution'])
        plt.figure(figsize=(max(6, 0.3 * heat_table.shape[1]), max(4, 0.3 * heat_table.shape[0])))
        plt.imshow(heat_table, aspect='auto', cmap='Blues')
        plt.colorbar(label='Course count')
        plt.xticks(range(len(heat_table.columns)), heat_table.columns, rotation=45, ha='right', fontsize=7)
        plt.yticks(range(len(heat_table.index)), heat_table.index, fontsize=7)
        plt.title("Theme coverage by Institution")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "heatmap_themes_institutions.png"))
        plt.close()
    # Heatmap: SWEBOK KAs × Topic Clusters
    # For each course, assign cluster label (topics clustering) and SWEBOK KAs
    df_tmp = df.copy()
    df_tmp['Topic Cluster'] = labels_topics
    # Expand SWEBOK areas to rows
    rows_swebok = []
    for cl, kas in zip(df_tmp['Topic Cluster'], df_tmp['SWEBOK Areas']):
        for ka in kas:
            rows_swebok.append({"Cluster": cl, "KA": ka})
    if rows_swebok:
        swebok_df = pd.DataFrame(rows_swebok)
        heat2 = pd.crosstab(swebok_df['KA'], swebok_df['Cluster'])
        plt.figure(figsize=(max(6, 0.3 * heat2.shape[1]), max(4, 0.3 * heat2.shape[0])))
        plt.imshow(heat2, aspect='auto', cmap='Greens')
        plt.colorbar(label='Count')
        plt.xticks(range(len(heat2.columns)), heat2.columns, rotation=45, ha='right', fontsize=7)
        plt.yticks(range(len(heat2.index)), heat2.index, fontsize=7)
        plt.title("SWEBOK KA coverage by Topic Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "heatmap_swebok_clusters.png"))
        plt.close()


def write_csvs(df_clean: pd.DataFrame, df_topics: pd.DataFrame, topic_labels: np.ndarray, desc_labels: np.ndarray, diagnostics_topics: Dict[int, Dict[str, float]], diagnostics_desc: Dict[int, Dict[str, float]], swebok_map: Dict[str, List[str]], outdir: str) -> None:
    """Write multiple CSV outputs to the working directory.

    The following files are written:
    - courses_cleaned.csv: cleaned and normalised data.
    - topics_exploded.csv: long form topic data.
    - topic_model_results.csv: per course cluster labels and diagnostics.
    - swebok_mapping.csv: course to SWEBOK mapping with scores.

    Parameters
    ----------
    df_clean: DataFrame
        Cleaned data frame.
    df_topics: DataFrame
        Exploded topics table.
    topic_labels: ndarray
        Cluster labels for topics representation.
    desc_labels: ndarray
        Cluster labels for descriptions representation.
    diagnostics_topics: dict
        Diagnostics for topics clustering.
    diagnostics_desc: dict
        Diagnostics for description clustering.
    swebok_map: dict
        Mapping from canonical topic to SWEBOK KAs.
    outdir: str
        Directory where files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    # Save cleaned data
    df_clean.to_csv(os.path.join(outdir, "courses_cleaned.csv"), index=False)
    # Save exploded topics
    df_topics.to_csv(os.path.join(outdir, "topics_exploded.csv"), index=False)
    # Topic model results
    results = pd.DataFrame({
        "Course Index": range(len(topic_labels)),
        "Topic Cluster": topic_labels,
        "Description Cluster": desc_labels
    })
    # Expand diagnostics into readable columns
    diag_records = []
    for k, metrics in diagnostics_topics.items():
        diag_records.append({"k": k, "silhouette": metrics.get("silhouette", np.nan), "davies_bouldin": metrics.get("davies_bouldin", np.nan), "representation": "topics"})
    for k, metrics in diagnostics_desc.items():
        diag_records.append({"k": k, "silhouette": metrics.get("silhouette", np.nan), "davies_bouldin": metrics.get("davies_bouldin", np.nan), "representation": "descriptions"})
    diag_df = pd.DataFrame(diag_records)
    results_path = os.path.join(outdir, "topic_model_results.csv")
    # Write results; also attach diagnostics summary at bottom of file for reference
    results.to_csv(results_path, index=False)
    diag_df.to_csv(os.path.join(outdir, "topic_model_diagnostics.csv"), index=False)
    # SWEBOK mapping per course
    swebok_records = []
    for idx, areas, scores in zip(df_clean.index, df_clean["SWEBOK Areas"], df_clean["SWEBOK Scores"]):
        swebok_records.append({"Course Index": idx, "SWEBOK Areas": "; ".join(areas), "SWEBOK Scores": json.dumps(scores)})
    swebok_df = pd.DataFrame(swebok_records)
    swebok_df.to_csv(os.path.join(outdir, "swebok_mapping.csv"), index=False)


def generate_manifest(df_clean: pd.DataFrame, unique_topics: List[str], topic_labels: np.ndarray, outdir: str) -> str:
    """Generate a human‑readable manifest summarising the analysis.

    Parameters
    ----------
    df_clean: DataFrame
        Cleaned data frame.
    unique_topics: List[str]
        Sorted list of canonical topics.
    topic_labels: ndarray
        Cluster labels assigned to courses.
    outdir: str
        Directory containing output files.

    Returns
    -------
    str
        Manifest string.
    """
    n_courses = df_clean.shape[0]
    n_institutions = df_clean['Offering Institution'].nunique()
    n_topics = len(unique_topics)
    cluster_counts = pd.Series(topic_labels).value_counts().sort_index().to_dict()
    noise_pct = 0.0  # KMeans does not produce noise
    manifest_lines = [
        f"Number of courses analysed: {n_courses}",
        f"Number of unique institutions: {n_institutions}",
        f"Number of unique normalised topics: {n_topics}",
        f"Cluster size distribution (topic clusters): {cluster_counts}",
        f"Percentage of noise points: {noise_pct:.2%}",
    ]
    # File sizes
    files = ["courses_cleaned.csv", "topics_exploded.csv", "topic_model_results.csv", "topic_model_diagnostics.csv", "swebok_mapping.csv", "taxonomy.json"]
    for fname in files:
        fpath = os.path.join(outdir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            manifest_lines.append(f"{fname}: {size} bytes")
    return "\n".join(manifest_lines)


def main():
    """Run the entire analysis pipeline.

    Reads the input CSV ``Courses data.csv`` from the current working
    directory, applies cleaning and normalisation, performs topic
    modelling and clustering, builds a co‑occurrence network, maps
    topics to SWEBOK knowledge areas, generates figures and saves
    multiple output files.  At the end of execution, prints a
    manifest summarising key statistics.
    """
    input_csv = os.path.join(os.getcwd(), "Courses data.csv")
    if not os.path.exists(input_csv):
        print(f"Input file '{input_csv}' not found.")
        return
    df_raw = load_data(input_csv)
    df_clean, synonyms, inv_map = clean_normalize(df_raw)
    df_topics = explode_topics(df_clean)
    unique_topics = sorted(set(df_topics['Topic']))
    # Build matrices
    X_topics, vocab_topics = build_topic_matrix(df_clean, unique_topics)
    X_desc, vocab_desc = build_description_matrix(df_clean, inv_map)
    # Cluster courses
    labels_topics, diag_topics, best_k_topics, labels_desc, diag_desc, best_k_desc = cluster_courses(X_topics, X_desc)
    # Compute co‑occurrence matrix and communities
    co_matrix, co_topics = compute_cooccurrence(df_topics)
    topic_comm = detect_communities(co_matrix, n_clusters=4)
    # SWEBOK mapping
    swebok_map = compute_swebok_mapping(unique_topics)
    df_clean = assign_swebok(df_clean, swebok_map)
    # Theme assignment
    themes = assign_themes(unique_topics)
    # Save taxonomy
    taxonomy_path = os.path.join(os.getcwd(), "taxonomy.json")
    save_taxonomy_json(unique_topics, synonyms, themes, swebok_map, taxonomy_path)
    # Figures
    figs_dir = os.path.join(os.getcwd(), "figs")
    make_figures(df_clean, df_topics, co_matrix, co_topics, topic_comm, labels_topics, labels_desc, swebok_map, themes, figs_dir)
    # Outputs CSVs
    output_dir = os.getcwd()
    write_csvs(df_clean, df_topics, labels_topics, labels_desc, diag_topics, diag_desc, swebok_map, output_dir)
    # Manifest
    manifest = generate_manifest(df_clean, unique_topics, labels_topics, output_dir)
    print(manifest)


if __name__ == "__main__":
    # Silence any warnings for a clean output
    warnings.simplefilter("ignore")
    main()