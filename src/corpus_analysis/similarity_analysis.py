#!/usr/bin/env python3
"""
Similarity Analysis Module

This module provides functions to evaluate the similarity between reference and
generated texts using various metrics such as BLEU, METEOR, ROUGE, and Sentence-BERT
cosine similarity. It processes all files in the specified directories and compiles
the results into a Pandas DataFrame for analysis.
"""

# --- Third-Party Imports ---
import mauve
from bisect import bisect_left
from scipy.stats import entropy
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import load_complete_corpus
from ..imports import (
    os,
    np,
    pd,
    nltk,
    torch,
    Counter,
    Dict,
    Optional,
    ABBREVIATIONS_MAPPING,
    DEVICE,
    RANDOM_SEED
)


def initialize_similarity_models():
    """
    Initializes models for S-BERT and ROUGE to avoid reloading them for every file.

    Returns:
        sbert_model: The loaded Sentence-BERT model for embedding generation.
        bert_model: The loaded BERT model for BERTScore calculations.
        bert_tokenizer: The tokenizer corresponding to the BERT model.
        rouge_scorer_instance: The initialized ROUGE scorer for ROUGE metric calculations.
    """
    logger.info(f"Loading models on device: {DEVICE}...")

    # Initialize Sentence-BERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

    # Initialize BERT model and tokenizer for BERTScore
    bert_tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/bsc-bio-ehr-es")
    bert_model = AutoModel.from_pretrained("PlanTL-GOB-ES/bsc-bio-ehr-es")
    bert_model = bert_model.to(DEVICE)

    # Initialize ROUGE scorer (1, 2, L)
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    return sbert_model, bert_model, bert_tokenizer, rouge_scorer_instance


def calculate_bleu(
    reference_text: str,
    generated_text: str
) -> float:
    """
    Calculates BLEU score with smoothing.

    Args:
        reference_text: The reference text to compare against.
        generated_text: The generated text to evaluate.

    Returns:
        BLEU score as a float between 0 and 1.
    """
    # Tokenize the texts
    reference_tokens = nltk.word_tokenize(reference_text.lower(), language="spanish")
    generated_tokens = nltk.word_tokenize(generated_text.lower(), language="spanish")

    # Using smoothing function to handle short texts better
    chencherry = SmoothingFunction()

    # Calculate BLEU score (using unigram BLEU for simplicity, can be extended to n-grams)
    bleu_score_value = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=chencherry.method4)
    return bleu_score_value


def calculate_meteor(
    reference_text: str,
    generated_text: str
) -> float:
    """
    Calculates METEOR score.

    Args:
        reference_text: The reference text to compare against.
        generated_text: The generated text to evaluate.

    Returns:
        METEOR score as a float between 0 and 1.
    """
    # Tokenize the texts
    reference_tokens = nltk.word_tokenize(reference_text.lower(), language="spanish")
    generated_tokens = nltk.word_tokenize(generated_text.lower(), language="spanish")

    # Calculate METEOR score
    meteor_score_value = meteor_score([reference_tokens], generated_tokens)
    return meteor_score_value


def calculate_rouge(
    reference_text: str,
    generated_text: str,
    rouge_scorer: rouge_scorer.RougeScorer
) -> Dict[str, float]:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L.

    Args:
        reference_text: The reference text to compare against.
        generated_text: The generated text to evaluate.
        rouge_scorer: An initialized ROUGE scorer instance to use for scoring.

    Returns:
        A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    # Calculate ROUGE scores using the initialized scorer
    scores = rouge_scorer.score(reference_text, generated_text)
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }


def calculate_sbert_similarity(
    reference_text: str,
    generated_text: str,
    sbert_model: SentenceTransformer
) -> float:
    """
    Calculates Cosine Similarity using Sentence-BERT embeddings.
    
    Args:
        reference_text: The reference text to compare against.
        generated_text: The generated text to evaluate.
        sbert_model: An initialized SentenceTransformer model to use for encoding.

    Returns:
        Cosine similarity score as a float between 0 and 1.
    """
    # Generate embeddings for both texts using the provided Sentence-BERT model
    embeddings = sbert_model.encode([reference_text, generated_text], convert_to_tensor=True)

    # Calculate cosine similarity between the two embeddings
    cosine_similarity_value = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_similarity_value


def calculate_tfidf_cosine(
    reference_text: str,
    generated_text: str
) -> float:
    """
    Calculates traditional Cosine Similarity using TF-IDF.

    Args:
        reference_text: The reference text to compare against.
        generated_text: The generated text to evaluate.

    Returns:
        Cosine similarity score as a float between 0 and 1. 
    """
    # Create TF-IDF vectors for both texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference_text, generated_text])

    # Calculate cosine similarity between the two vectors
    cosine_similarity_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_similarity_value


def get_ngram_profile(
    texts: list[str],
    n: int = 4,
    top_k: int = 5000
) -> tuple[np.ndarray, list[str]]:
    """
    Build frequency vector for an entire corpus using top-k most frequent n-grams.
    
    Args:
        texts: List of all documents in the corpus.
        n: The length of n-grams to use for the analysis (default is 4).
        top_k: The number of top n-grams to consider for the frequency vector (default is 5000).

    Returns:
        A tuple containing:
        - freq_vector: A numpy array representing the frequency of the top-k n-grams.
        - vocab: A list of the top-k n-grams corresponding to the frequency vector.
    """
    # Count n-grams across the entire corpus
    total_counter = Counter()
    
    for text in texts:
        # Generate n-grams for the current text
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]

        # Update the total counter with the n-grams from this text
        current_counter = Counter(ngrams)
        total_counter.update(current_counter)
    
    # Take the top_k most common n-grams as our vocabulary
    vocab = [ngram for ngram, _ in total_counter.most_common(top_k)]
    freq_vector = np.zeros(len(vocab))
    
    # Fill the frequency vector based on the total counts of the top-k n-grams
    for i, ngram in enumerate(vocab):
        freq_vector[i] = total_counter[ngram]
    
    # Normalize to probability distribution (important!)
    freq_vector = freq_vector / freq_vector.sum() if freq_vector.sum() > 0 else freq_vector
    
    return freq_vector, vocab


def calculate_ngams_cosine(
    reference_texts: list[str],
    generated_texts: list[str],
    n: int = 4,
    top_k: int = 5000
) -> float:
    """
    Calculates Cosine Similarity based on character n-gram frequency vectors.

    Args:
        reference_texts: The list of reference texts to compare against.
        generated_texts: The list of generated texts to evaluate.
        n: The length of n-grams to use for the analysis (default is 4).
        top_k: The number of top n-grams to consider for the frequency vector (default is 5000).
    
    Returns:
        Cosine similarity score as a float between 0 and 1.
    """
    vec1, vocab1 = get_ngram_profile(reference_texts, n=n, top_k=top_k)
    vec2, vocab2 = get_ngram_profile(generated_texts, n=n, top_k=top_k)
    
    # Use the union of both vocabularies to align vectors
    all_ngrams = sorted(set(vocab1) | set(vocab2))
    v1 = np.zeros(len(all_ngrams))
    v2 = np.zeros(len(all_ngrams))
    
    for i, ng in enumerate(all_ngrams):
        if ng in vocab1:
            v1[i] = vec1[vocab1.index(ng)]
        if ng in vocab2:
            v2[i] = vec2[vocab2.index(ng)]
    
    # Normalize again
    v1 = v1 / v1.sum() if v1.sum() > 0 else v1
    v2 = v2 / v2.sum() if v2.sum() > 0 else v2
    
    # Similarity metrics
    cosine_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
    hellinger = np.sqrt(np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2)) / np.sqrt(2)

    # Compute the mixture distribution M
    m = 0.5 * (v1 + v2)

    # JSD is the average KL divergence of v1 and v2 from M
    js_divergence = 1 - (0.5 * entropy(v1 + 1e-12, m + 1e-12) + 0.5 * entropy(v2 + 1e-12, m + 1e-12))

    return {
        "cosine": cosine_sim,
        "1_js_divergence": js_divergence,
        "hellinger": hellinger
    }


def get_embeddings(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> torch.Tensor:
    """
    Generate token embeddings for the input text using BERT.

    Args:
        text: Input text or batch of sentences.
        tokenizer: The tokenizer corresponding to the BERT model.
        model: The BERT model to use for generating embeddings.

    Returns:
        torch.Tensor: Token embeddings with shape (batch_size, seq_len, hidden_dim).
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to GPU if available
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Compute embeddings without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Return last hidden states (token-level embeddings)
    # Asegurarse de que sea un tensor de PyTorch
    embeddings = outputs.last_hidden_state
    
    return embeddings


def bert_cosine_similarity(
    generated_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        generated_embeddings: Embeddings of candidate tokens.
        reference_embeddings: Embeddings of reference tokens.

    Returns:
        torch.Tensor: Cosine similarity matrix.
    """
    # Asegurarse de que son tensores de PyTorch
    if not isinstance(generated_embeddings, torch.Tensor):
        generated_embeddings = torch.tensor(generated_embeddings.toarray() if hasattr(generated_embeddings, 'toarray') else generated_embeddings)
    
    if not isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = torch.tensor(reference_embeddings.toarray() if hasattr(reference_embeddings, 'toarray') else reference_embeddings)
    
    # Mover a la misma ubicación si es necesario
    generated_embeddings = generated_embeddings.to(DEVICE)
    reference_embeddings = reference_embeddings.to(DEVICE)
    
    # Normalize embeddings along the hidden dimension
    generated_embeddings = torch.nn.functional.normalize(generated_embeddings, p=2, dim=-1)
    reference_embeddings = torch.nn.functional.normalize(reference_embeddings, p=2, dim=-1)

    # Compute similarity using batched matrix multiplication
    return torch.bmm(generated_embeddings, reference_embeddings.transpose(1, 2))


def bert_score(
    candidate: str,
    reference: str,
    bert_tokenizer: AutoTokenizer,
    bert_model: AutoModel
) -> Dict[str, float]:
    """
    Compute BERTScore (Precision, Recall, F1) between a candidate and a reference sentence.

    Args:
        candidate: Candidate sentence.
        reference: Reference sentence.
        bert_tokenizer: The tokenizer corresponding to the BERT model.
        bert_model: The BERT model to use for generating embeddings.

    Returns:
        dict: Dictionary containing precision, recall, and F1 scores.
    """
    # Get token embeddings for candidate and reference
    candidate_embeddings = get_embeddings(candidate, bert_tokenizer, bert_model)
    reference_embeddings = get_embeddings(reference, bert_tokenizer, bert_model)

    # Compute cosine similarity matrix
    similarity_matrix = bert_cosine_similarity(candidate_embeddings, reference_embeddings)

    # Calculate precision, recall, and F1 scores
    precision = similarity_matrix.max(dim=2)[0].mean()
    recall = similarity_matrix.max(dim=1)[0].mean()

    if precision + recall == 0:
        f1_score = torch.tensor(0.0)
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Return scores as a dictionary
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
    }


def tokenize_text(text: str) -> list[str]:
    """
    Lowercases and tokenizes a text with the same tokenizer used by BLEU and METEOR.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase word tokens.
    """
    return nltk.word_tokenize(text.lower(), language="spanish")


def subsample_to_budget(
    token_lists: list[list[str]],
    budget: Optional[int],
    seed: int = RANDOM_SEED
) -> list[list[str]]:
    """
    Pick whole documents at random until `budget` tokens are reached (None = keep all).
    
    Args:
        token_lists: List of lists of tokens.
        budget: Maximum number of tokens to include (None = keep all).
        seed: Random seed for reproducibility.

    Returns:
        List of lists of tokens, subsampled to the given budget.
    """
    if budget is None:
        return token_lists
    rng = np.random.default_rng(seed)
    out, total = [], 0
    for i in rng.permutation(len(token_lists)):
        out.append(token_lists[i]); total += len(token_lists[i])
        if total >= budget:
            break
    return out


def _ngrams(tokens: list[str], n: int):
    """Yields the n-grams (as tuples) of a token list."""
    return zip(*(tokens[i:] for i in range(n)))


def calculate_self_bleu(
    texts: list[str],
    max_n: int = 4,
    n_samples: int = 500,
    epsilon: float = 0.1
) -> Dict[str, float]:
    """
    Calculate Self-BLEU for a list of texts.
    
    Self-BLEU: each document vs. ALL OTHER documents of the same corpus (lower = more diverse).
    Uses an n-gram index with top-2 counts, so it is linear instead of quadratic.
    Smoothing equals NLTK method1.
    
    Args:
        texts: List of texts to evaluate.
        max_n: Maximum n-gram length to consider (default is 4).
        n_samples: Number of samples to evaluate (default is 500).
        epsilon: Smoothing parameter (default is 0.1).
    
    Returns:
        Dictionary containing Self-BLEU score and its standard deviation.
    """
    docs = [tokenize_text(t) for t in texts if t and t.strip()]
    lengths = np.array([len(d) for d in docs])
    order = np.argsort(lengths)
    sorted_len = lengths[order]
    sorted_len_list = sorted_len.tolist()

    doc_counts = [[Counter(_ngrams(d, n)) for n in range(1, max_n + 1)] for d in docs]
    index = [defaultdict(lambda: [0, -1, 0]) for _ in range(max_n)]   # [best, best_doc, second]
    for di, per_n in enumerate(doc_counts):
        for k, cnt in enumerate(per_n):
            for g, c in cnt.items():
                e = index[k][g]
                if c > e[0]:
                    e[2], e[0], e[1] = e[0], c, di
                elif c > e[2]:
                    e[2] = c
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(docs))
    if n_samples and n_samples < len(docs):
        idx = rng.choice(idx, size=n_samples, replace=False)

    scores = []
    for i in idx:
        hyp_len = len(docs[i])
        if hyp_len == 0:
            continue
        logp = 0.0
        for k in range(max_n):
            num = den = 0
            for g, c in doc_counts[i][k].items():
                best, who, second = index[k][g]
                num += min(c, second if who == i else best)
                den += c
            if den == 0:
                logp = None
                break
            p = num / den if num > 0 else epsilon / den
            logp += np.log(p) / max_n
        if logp is None:
            continue
        pos = bisect_left(sorted_len_list, hyp_len)          # closest other-document length
        cands = [sorted_len[j] for j in (pos - 2, pos - 1, pos, pos + 1)
                 if 0 <= j < len(order) and order[j] != i]
        ref_len = min(cands, key=lambda r: (abs(r - hyp_len), r))
        bp = 1.0 if hyp_len > ref_len else np.exp(1 - ref_len / hyp_len)
        scores.append(bp * np.exp(logp))
    return {f"Self_BLEU_{max_n}": float(np.mean(scores)),
            f"Self_BLEU_{max_n}_std": float(np.std(scores))}


def calculate_lexical_distribution(
    texts: list[str],
    token_budget: Optional[int] = None,
    n_repeats: int = 5,
    zipf_max_rank: int = 1000,
) -> Dict[str, float]:
    """
    Calculate lexical distribution metrics for a list of texts.
    
    Unigram/bigram entropy (bits), distinct-1/2/3, Zipf exponent (log-log OLS over ranks
    1..zipf_max_rank) and share of tokens covered by the 100 most frequent word types.
    
    Args:
        texts: List of texts to analyze.
        token_budget: Maximum number of tokens to include (None = keep all).
        n_repeats: Number of times to repeat the analysis.
        zipf_max_rank: Maximum rank for Zipf analysis.

    Returns:
        A dictionary containing the calculated metrics.
    """
    tokenized = [tokenize_text(t) for t in texts if t and t.strip()]
    reps = n_repeats if token_budget else 1
    rows = []
    for r in range(reps):
        docs = subsample_to_budget(tokenized, token_budget, RANDOM_SEED + r)
        flat = [w for d in docs for w in d]
        uni = Counter(flat)
        bi = Counter(g for d in docs for g in _ngrams(d, 2))

        def H(cnt):
            p = np.array(list(cnt.values()), dtype=float); p /= p.sum()
            return float(-(p * np.log2(p)).sum())

        row = {"Entropy_unigram": H(uni), "Entropy_bigram": H(bi)}
        for n in (1, 2, 3):
            grams = [g for d in docs for g in _ngrams(d, n)]
            row[f"Distinct_{n}"] = len(set(grams)) / max(len(grams), 1)
        freqs = np.array(sorted(uni.values(), reverse=True), dtype=float)
        top = min(zipf_max_rank, len(freqs))
        lr, lf = np.log(np.arange(1, top + 1)), np.log(freqs[:top])
        slope, intercept = np.polyfit(lr, lf, 1)
        pred = slope * lr + intercept
        row["Zipf_alpha"] = float(-slope)
        row["Zipf_R2"] = float(1 - ((lf - pred) ** 2).sum() / ((lf - lf.mean()) ** 2).sum())
        row["Top100_mass"] = float(freqs[:100].sum() / freqs.sum())
        row["Vocab_size"], row["Tokens"] = len(uni), len(flat)
        rows.append(row)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def chunk_text(
    text: str,
    max_words: int = 100
) -> list[str]:
    """
    Splits a text into consecutive chunks of at most `max_words` words.

    Args:
        text: Text to split.
        max_words: Maximum number of words per chunk.

    Returns:
        List of text chunks.
    """
    w = text.split()
    return [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]


def calculate_mauve(
    reference_texts: list[str],
    generated_texts: list[str],
    embedder: SentenceTransformer,
    max_samples: int = 5000,
    max_words: int = 100,
    seeds: tuple = (0, 1, 2, 3, 4),
    device_id: int = 0
) -> Dict[str, float]:
    """
    Calculates MAUVE (reference = p, generated = q) averaged over several seeds.

    Texts are split into ~`max_words`-word chunks to get enough samples, and both
    sample sizes are equalised.

    Args:
        reference_texts: The list of reference texts.
        generated_texts: The list of generated texts.
        embedder: A multilingual SentenceTransformer used to embed the chunks.
        max_samples: Maximum number of chunks per distribution.
        max_words: Words per chunk.
        seeds: Seeds to average over.
        device_id: GPU id for the clustering (-1 runs it on CPU).

    Returns:
        Dictionary with the mean MAUVE score and its standard deviation.
    """
    p = [c for t in reference_texts for c in chunk_text(t, max_words)]
    q = [c for t in generated_texts for c in chunk_text(t, max_words)]
    vals = []
    for s in seeds:
        rng = np.random.default_rng(RANDOM_SEED + s)
        n = min(len(p), len(q), max_samples)
        pi, qi = rng.choice(len(p), n, replace=False), rng.choice(len(q), n, replace=False)
        pf = embedder.encode([p[i] for i in pi], batch_size=64, show_progress_bar=False)
        qf = embedder.encode([q[i] for i in qi], batch_size=64, show_progress_bar=False)
        vals.append(mauve.compute_mauve(p_features=pf, q_features=qf, seed=s,
                                        device_id=device_id, verbose=False).mauve)
    return {"MAUVE": float(np.mean(vals)), "MAUVE_std": float(np.std(vals))}


def print_similarity_report(
    data_path: str,
    prompts: list[str],
    models: list[str],
    output_path: Optional[str] = None,
    token_budget: Optional[int] = 100000
):
    """
    Evaluates the similarity between reference and generated texts for all files in 
    the specified directories and compiles the results into a DataFrame.

    Args:
        data_path: The base path where the corpora are located.
        prompts: List of prompt methods to evaluate (e.g., ["method_1_var", "method_3_context"]).
        models: List of models to evaluate (e.g., ["llama3.2-3b", "gpt-4.1-nano"]).
        output_path: Optional path to save the resulting DataFrame as a CSV file.
        token_budget: Optional token budget used to equalise corpora in the lexical distribution metrics.
                      Default is 100,000 tokens, because the original corpus has ~100k tokens.
    """
    # Define paths and names for corpora to compare
    base_path = "raw/train"
    base_corpus = load_complete_corpus(data_path=os.path.join(data_path, base_path))
    mauve_embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=DEVICE)

    # Initialize models once for efficiency
    sbert_model, bert_model, bert_tokenizer, rouge_scorer_instance = initialize_similarity_models()
    general_results = pd.DataFrame([])

    # Corpus-level diversity metrics of the original corpus (computed once)
    base_texts = [d["clean_text"] for d in base_corpus.values() if d.get("clean_text")]
    original_row = {
        "Filename": "ORIGINAL", "Model": "Original", "Method": "train",
        **calculate_self_bleu(base_texts),
        **calculate_lexical_distribution(base_texts, token_budget=token_budget)
    }
    general_results = pd.concat([general_results, pd.DataFrame([original_row])], ignore_index=True)

    for model in models:
        for prompt in prompts:
            # Get abbreviations for prompt and model
            prompt_abbr = ABBREVIATIONS_MAPPING.get(prompt, prompt)
            model_abbr = model.replace(".", "")

            # Load the augmented corpus for this model and prompt
            augmented_corpus_path = f"corrected/{prompt_abbr}/{model_abbr}"
            augmented_corpus = load_complete_corpus(data_path=os.path.join(data_path, augmented_corpus_path))

            logger.info(f"Found {len(base_corpus)} documents. Starting evaluation for model '{model}' and prompt '{prompt}'...")
            augmented_results = []

            # Reference and generated texts of this model/prompt (for corpus-level metrics)
            reference_texts = []
            generated_texts = []

            # Calculate similarity metrics for each document against the corresponding generated one
            for i, filename in enumerate(list(base_corpus.keys())):
                logger.info(f"Processing {filename} ({i+1}/{len(base_corpus)})...")

                # Get the clean text for both reference and generated documents
                reference_text = base_corpus[filename]["clean_text"]
                generated_text = augmented_corpus.get(filename, {}).get("clean_text", "")

                if not generated_text:
                    logger.warning(f"No generated text found for {filename}. Skipping metrics calculation.")
                    continue

                # Append texts to the lists for corpus-level metrics calculated later
                reference_texts.append(reference_text)
                generated_texts.append(generated_text)
                
                tfidf_cos = meteor = bleu = sbert_cos = 0.0
                rouge_scores = {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
                sbert_cos = bert_scores = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

                # BLEU
                bleu = calculate_bleu(reference_text, generated_text)
                
                # METEOR
                meteor = calculate_meteor(reference_text, generated_text)
                
                # TF-IDF Cosine
                tfidf_cos = calculate_tfidf_cosine(reference_text, generated_text)
                
                # ROUGE
                rouge_scores = calculate_rouge(reference_text, generated_text, rouge_scorer_instance)
                
                # Sentence-BERT
                sbert_cos = calculate_sbert_similarity(reference_text, generated_text, sbert_model)

                # BERTScore
                bert_scores = bert_score(generated_text, reference_text, bert_tokenizer, bert_model)

                augmented_results.append({
                    "Filename": filename,
                    "Model": model,
                    "Method": prompt,
                    "Cosine_Sim_TFIDF": tfidf_cos,
                    "METEOR": meteor,
                    "BLEU": bleu,
                    "ROUGE-1": rouge_scores["ROUGE-1"],
                    "ROUGE-2": rouge_scores["ROUGE-2"],
                    "ROUGE-L": rouge_scores["ROUGE-L"],
                    "SBERT_Cosine": sbert_cos,
                    "BERTScore_Precision": bert_scores["precision"],
                    "BERTScore_Recall": bert_scores["recall"],
                    "BERTScore_F1": bert_scores["f1_score"]
                })

            # Create DataFrame
            augmented_results = pd.DataFrame(augmented_results)
            
            # Calculate n-gram cosine similarity for the entire corpus
            ngram_cosine_results = calculate_ngams_cosine(reference_texts, generated_texts, n=4, top_k=5000)
            logger.info(f"Calculated n-gram cosine similarity for model '{model}' and prompt '{prompt}': {ngram_cosine_results}")

            # Add n-gram cosine similarity results to the DataFrame (same value for all rows since it's corpus-level)
            augmented_results["Cosine_Sim_NGram"] = ngram_cosine_results["cosine"]
            augmented_results["1-JSD"] = ngram_cosine_results["1_js_divergence"]
            augmented_results["Hellinger_Distance"] = ngram_cosine_results["hellinger"]

            # Corpus-level diversity metrics (same value for all rows)
            corpus_metrics = {
                **calculate_self_bleu(generated_texts),
                **calculate_lexical_distribution(generated_texts, token_budget=token_budget),
                **calculate_mauve(reference_texts, generated_texts, mauve_embedder)
            }
            logger.info(f"Calculated corpus-level metrics for model '{model}' and prompt '{prompt}': {corpus_metrics}")
            for metric_name, metric_value in corpus_metrics.items():
                augmented_results[metric_name] = metric_value
            
            # Calculate Averages
            means = augmented_results.mean(numeric_only=True)
            means["Filename"] = "AVERAGE"
            means["Model"] = model
            means["Method"] = prompt

            # Calculate standard deviations for referencee
            stds = augmented_results.std(numeric_only=True)
            stds["Filename"] = "STD_DEV"
            stds["Model"] = model
            stds["Method"] = prompt
            
            # Concatenate the average and standard deviation rows to the results DataFrame
            augmented_results = pd.concat([augmented_results, pd.DataFrame([means])], ignore_index=True)
            augmented_results = pd.concat([augmented_results, pd.DataFrame([stds])], ignore_index=True)

            # Print the results for this model and prompt
            logger.info(f"Similarity Report for model '{model}' and prompt '{prompt}':")
            logger.info(augmented_results.round(4))

            if output_path:
                file_path = os.path.join(output_path, f"{prompt_abbr}/similarity_report_{model_abbr}_{prompt_abbr}.csv")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Save the results to a CSV file for this model and prompt
                augmented_results.to_csv(file_path, index=False, sep=";")
                logger.info(f"Saved similarity report for model '{model}' and prompt '{prompt}' to {output_path}")

            # Append average results to general results for overall comparison
            general_results = pd.concat([general_results, pd.DataFrame([means]), pd.DataFrame([stds])], ignore_index=True)
        
    # Remove the "Filename" column from the general results as it only contains "AVERAGE"
    general_results = general_results.drop(columns=["Filename"])

    # Reorder columns to have "Model" and "Method" first
    cols = ["Model", "Method"] + [col for col in general_results.columns if col not in ["Model", "Method"]]
    general_results = general_results[cols]

    if output_path:
        # Save the overall results to a CSV file
        general_results.to_csv(os.path.join(output_path, "similarity_report_all_models.csv"), index=False, sep=";")
        logger.info(f"Saved overall similarity report for all models and prompts to {output_path}")
    
    # Print the results for this model and prompt
    logger.info(f"Similarity Report for model '{model}' and prompt '{prompt}':")
    logger.info(general_results.round(4))
