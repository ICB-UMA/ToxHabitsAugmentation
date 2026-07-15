#!/usr/bin/env python3
"""
Similarity Analysis Module

This module provides functions to evaluate the similarity between reference and
generated texts using various metrics such as BLEU, METEOR, ROUGE, and Sentence-BERT
cosine similarity. It processes all files in the specified directories and compiles
the results into a Pandas DataFrame for analysis.
"""

# --- Third-Party Imports ---
from scipy.stats import entropy
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
    DEVICE
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
    js_divergence = entropy(v1 + 1e-12, v2 + 1e-12) / 2 + entropy(v2 + 1e-12, v1 + 1e-12) / 2
    hellinger = np.sqrt(np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2)) / np.sqrt(2)

    return {
        "cosine": cosine_sim,
        "js_divergence": js_divergence,
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


def print_similarity_report(
    data_path: str,
    prompts: list[str],
    models: list[str],
    output_path: Optional[str] = None
):
    """
    Evaluates the similarity between reference and generated texts for all files in 
    the specified directories and compiles the results into a DataFrame.

    Args:
        data_path: The base path where the corpora are located.
        prompts: List of prompt methods to evaluate (e.g., ["method_1_var", "method_3_context"]).
        models: List of models to evaluate (e.g., ["llama3.2-3b", "gpt-4.1-nano"]).
        output_path: Optional path to save the resulting DataFrame as a CSV file.
    """
    # Define paths and names for corpora to compare
    base_path = "raw/train"
    base_corpus = load_complete_corpus(data_path=os.path.join(data_path, base_path))

    # Initialize models once for efficiency
    sbert_model, bert_model, bert_tokenizer, rouge_scorer_instance = initialize_similarity_models()
    general_results = pd.DataFrame([])

    # Initialize lists to store reference and generated texts
    reference_texts = []
    generated_texts = []

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

            # Calculate similarity metrics for each document against the corresponding generated one
            for i, filename in enumerate(list(base_corpus.keys())[:10]):
                logger.info(f"Processing {filename} ({i+1}/{len(base_corpus)})...")

                # Get the clean text for both reference and generated documents
                reference_text = base_corpus[filename]["clean_text"]
                generated_text = augmented_corpus.get(filename, {}).get("clean_text", "")

                # Append texts to the lists for n-gram cosine similarity calculation later
                reference_texts.append(reference_text)
                generated_texts.append(generated_text)

                if not generated_text:
                    logger.warning(f"No generated text found for {filename}. Skipping metrics calculation.")
                    continue
                
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
            augmented_results["JS_Divergence"] = ngram_cosine_results["js_divergence"]
            augmented_results["Hellinger_Distance"] = ngram_cosine_results["hellinger"]
            
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
