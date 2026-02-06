#!/usr/bin/env python3
"""
Similarity Analysis Module

This module provides functions to evaluate the similarity between reference and
generated texts using various metrics such as BLEU, METEOR, ROUGE, and Sentence-BERT
cosine similarity. It processes all files in the specified directories and compiles
the results into a Pandas DataFrame for analysis.
"""

# --- Third-Party Imports ---
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import load_complete_corpus
from ..imports import (
    os,
    pd,
    nltk,
    Dict,
    Optional,
    DEVICE
)


def initialize_similarity_models():
    """
    Initializes models for S-BERT and ROUGE to avoid reloading them for every file.

    Returns:
        sbert_model: The loaded Sentence-BERT model for embedding generation.
        rouge_scorer_instance: The initialized ROUGE scorer for ROUGE metric calculations.
    """
    logger.info(f"Loading models on device: {DEVICE}...")

    # Initialize Sentence-BERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

    # Initialize ROUGE scorer (1, 2, L)
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    return sbert_model, rouge_scorer_instance


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
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    generated_tokens = nltk.word_tokenize(generated_text.lower())

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
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    generated_tokens = nltk.word_tokenize(generated_text.lower())

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
        prompts: List of prompt methods to evaluate (e.g., ["method_1_var", "method_2_context"]).
        models: List of models to evaluate (e.g., ["llama3.2-3b", "gpt-4.1-nano"]).
        output_path: Optional path to save the resulting DataFrame as a CSV file.
    """
    # Define paths and names for corpora to compare
    base_path = "raw/train"
    base_corpus = load_complete_corpus(data_path=os.path.join(data_path, base_path))

    # Initialize models once for efficiency
    sbert_model, rouge_scorer_instance = initialize_similarity_models()
    general_results = pd.DataFrame([])

    for model in models:
        for prompt in prompts:
            # Load the augmented corpus for this model and prompt
            augmented_corpus_path = f"augmented/{prompt}/{model}"
            augmented_corpus = load_complete_corpus(data_path=os.path.join(data_path, augmented_corpus_path))

            logger.info(f"Found {len(base_corpus)} documents. Starting evaluation for model '{model}' and prompt '{prompt}'...")
            augmented_results = []

            # Calculate similarity metrics for each document against the corresponding generated one
            for i, filename in enumerate(list(base_corpus.keys())[:10]):
                logger.info(f"Processing {filename} ({i+1}/{len(base_corpus)})...")

                # Get the clean text for both reference and generated documents
                reference_text = base_corpus[filename]["clean_text"]
                generated_text = augmented_corpus.get(filename, {}).get("clean_text", "")

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

                augmented_results.append({
                    "Filename": filename,
                    "Model": model,
                    "Method": prompt,
                    "BLEU": bleu,
                    "METEOR": meteor,
                    "Cosine_Sim_TFIDF": tfidf_cos,
                    "ROUGE-1": rouge_scores["ROUGE-1"],
                    "ROUGE-2": rouge_scores["ROUGE-2"],
                    "ROUGE-L": rouge_scores["ROUGE-L"],
                    "SBERT_Cosine": sbert_cos
                })

            # Create DataFrame
            augmented_results = pd.DataFrame(augmented_results)
            
            # Calculate Averages
            means = augmented_results.mean(numeric_only=True)
            means["Filename"] = "AVERAGE"
            means["Model"] = model
            means["Method"] = prompt
            augmented_results = pd.concat([augmented_results, pd.DataFrame([means])], ignore_index=True)

            # Print the results for this model and prompt
            logger.info(f"Similarity Report for model '{model}' and prompt '{prompt}':")
            logger.info(augmented_results.round(4))

            if output_path:
                file_path = os.path.join(output_path, f"{prompt}/similarity_report_{model}_{prompt}.csv")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Save the results to a CSV file for this model and prompt
                augmented_results.to_csv(file_path, index=False, sep=";")
                logger.info(f"Saved similarity report for model '{model}' and prompt '{prompt}' to {output_path}")

            # Append average results to general results for overall comparison
            general_results = pd.concat([general_results, pd.DataFrame([means])], ignore_index=True)
        
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
