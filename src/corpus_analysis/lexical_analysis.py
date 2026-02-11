#!/usr/bin/env python3
"""
Lexical analysis utilities for the ToxHabitsAugmentation project.

This module provides functions to perform comprehensive linguistic
analysis on a corpus of documents, including word and sentence counts,
average lengths, frequency distributions, negation cue analysis, misspelling
detection, and lexical diversity metrics.
"""

# --- Third-Party Imports ---
from spellchecker import SpellChecker

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import load_complete_corpus
from ..imports import (
    os,
    np,
    pd,
    re,
    plt,
    json,
    nltk,
    Any,
    Dict,
    Path,
    Counter,
    Optional
)


# --- Constant Definitions ---
# Define common negation cues in Spanish
COMMON_NEGATION_CUES = [
    "no", "nunca", "nadie", "nada", "ninguno", "ninguna", "jamás", "ni", "tampoco", "sin",
    "siquiera", "ni siquiera", "no hay", "no tiene", "no existe"
]

# Define a list of common Spanish words that not are abbreviations
NOT_ABBREVIATIONS = [
    "abi", "ago", "agua", "al", "algo", "alta", "alto", "anal", "and", "ano", "ante", "arco", "area",
    "as", "asas", "asi", "asma", "así", "ayer", "año", "años", "aún", "baja", "bajo", "base", "bazo",
    "baño", "bien", "boca", "bote", "buen", "cabe", "cada", "cama", "capa", "cara", "card", "care",
    "casi", "caso", "cava", "cede", "cel", "cell", "cena", "cita", "cito", "cola", "coli", "como",
    "comp", "con", "cono", "cook", "cree", "creo", "cruz", "cual", "cuya", "cuyo", "cuña", "da", "dada",
    "dado", "dar", "data", "days", "de", "debe", "dedo", "deja", "dejo", "del", "dia", "dias", "dic",
    "dice", "diez", "dos", "doy", "dres", "duda", "dura", "duro", "día", "días", "eco", "edad", "eje",
    "ejes", "el", "ella", "ello", "emma", "en", "es", "esa", "ese", "est", "esta", "este", "está", "esté",
    "exon", "exón", "fase", "feb", "fin", "foco", "for", "fosa", "frio", "fría", "frío", "fue", "fuga",
    "gain", "gas", "gen", "gene", "gran", "guía", "ha", "hace", "haga", "hago", "halo", "han", "hay",
    "hija", "hijo", "hoja", "hora", "hoy", "hoz", "háb", "ia", "ii", "iia", "iib", "iic", "iii", "iiia",
    "iiib", "iiic", "in", "ir", "iv", "iva", "ivb", "ivc", "juan", "jul", "jun", "ki", "kit", "la", "lado",
    "lago", "las", "le", "lee", "leve", "liga", "lo", "los", "loss", "lung", "luz", "mal", "mala", "mama",
    "mar", "mas", "masa", "may", "mayo", "me", "mes", "mesa", "mide", "modo", "moec", "must", "muy", "más",
    "nada", "neo", "ni", "nice", "no", "nos", "nota", "nov", "nula", "oct", "of", "ojal", "ok", "on",
    "once", "or", "oral", "osea", "oseo", "otra", "otro", "pala", "para", "paso", "pedi", "peor", "pero",
    "pese", "peso", "pico", "pido", "pies", "pio", "plan", "poco", "polo", "por", "port", "que", "raiz",
    "rama", "raíz", "real", "rita", "roce", "roja", "roux", "sabe", "saco", "sale", "san", "sana", "se",
    "sea", "seca", "seco", "seno", "sept", "ser", "será", "si", "sido", "sin", "sino", "site", "situ", "solo",
    "son", "stop", "su", "sus", "sí", "sólo", "test", "tipo", "toda", "todo", "toma", "top", "tos", "tras",
    "tres", "trás", "tubo", "tuvo", "type", "tía", "tío", "un", "una", "unas", "uno", "unos", "urea", "usa",
    "usar", "uso", "va", "van", "ve", "ven", "vena", "veo", "ver", "vez", "via", "vial", "vida", "vii", "vino",
    "viva", "vive", "vía", "vías", "wild", "with", "ya", "you", "zona", "área", "él", "éste", "ósea", "óseo"
]

# Spanish characters for regex
SPANISH_CHARS = "a-zA-ZñÑáéíóúÁÉÍÓÚüÜ"

# Regular expression to find common abbreviations (HTA, DM, E.C.G, etc.)
ABBREVIATION_PATTERN = re.compile(r'(?<![\w-])([' + SPANISH_CHARS + r']{1,2}(?:\.[' + SPANISH_CHARS + r']{1,2})*[' + SPANISH_CHARS + r']{1,2})(?![\w-])')


def perform_linguistic_analysis(
    corpus: Dict[str, Dict],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Performs a comprehensive linguistic analysis on a list of documents.

    Args:
        corpus: A dictionary where keys are document names and values
                contain entity data for each document.
        top_n: Number of top items to return for frequency-based metrics.

    Returns:
        A dictionary with various linguistic metrics and analyses.
    """
    if not corpus:
        # Handle empty corpus case
        return {
            "Total Documents": 0,
            "Total Word Count": 0,
            "Total Unique Words": 0,
            "Total Sentences": 0,
            "Avg. Sentence Length": 0.0,
            "Avg. Word Length": 0.0,
            f"Top {top_n} Words": [],
            f"Top {top_n} Abbreviations": [],
            "Negation Cue Frequency": 0.0,
            "Misspelling Rate": 0.0,
            "Lexical Diversity": 0.0,
        }

    # Initialize counters and lists
    all_words = []
    all_sentences = []
    abbreviations_found = []
    negation_cue_count = 0
    misspelled_words = []
    
    # Initialize spell checker
    spell = SpellChecker(language="es")
    logger.info("Starting linguistic analysis on corpus...")
    
    # Process each document
    for _, data in corpus.items():
        doc = data.get("clean_text", "").lower()

        # Sentence Tokenization
        sentences = nltk.tokenize.sent_tokenize(doc)
        all_sentences.extend(sentences)
        
        # Word Tokenization (for word counts and analysis)
        words = nltk.tokenize.word_tokenize(doc) # Use lower case for counting
        all_words.extend(words)

        # Negation Cue Frequency
        # Check for negation words in the tokenized list
        for word in words:
            if word.lower() in COMMON_NEGATION_CUES:
                negation_cue_count += 1
                
        # Abbreviations
        # Search the original, cased document for abbreviations using regex
        abbreviations = ABBREVIATION_PATTERN.findall(doc)

        # Filter out not abbreviations
        abbreviations = [abbr for abbr in abbreviations if abbr not in NOT_ABBREVIATIONS]
        abbreviations_found.extend(abbreviations)

        # Misspellings
        # Filter for alphabetic words to check against the spell checker"s dictionary
        alphabetic_words = [word for word in nltk.tokenize.word_tokenize(doc) if word.isalpha()]
        unknown_words = spell.unknown(alphabetic_words)
        misspelled_words.extend(unknown_words)
        
    # Filter out stopwords and punctuation for "Top Words" and vocabulary metrics
    stop_words = set(nltk.corpus.stopwords.words("spanish"))
    
    # Filter to get only alphabetic tokens, convert to lowercase, and exclude stopwords
    filtered_words_nonstop = [word for word in all_words if word not in stop_words]
    filtered_words_nonstop_nonnum = [word for word in all_words if word.isalpha() and word not in stop_words]
    
    total_words = len(all_words)
    total_sentences = len(all_sentences)
    total_documents = len(corpus)

    total_words_nonstop = len(filtered_words_nonstop)
    total_words_nonstop_nonnum = len(filtered_words_nonstop_nonnum)
    
    # Total Words and Unique Words
    unique_words = set(all_words)
    total_unique_words = len(unique_words)

    unique_words_nonstop = set(filtered_words_nonstop)
    total_unique_words_nonstop = len(unique_words_nonstop)

    unique_words_nonstop_nonnum = set(filtered_words_nonstop_nonnum)
    total_unique_words_nonstop_nonnum = len(unique_words_nonstop_nonnum)

    # Avg. Sentence Length
    sentence_lengths = [len(nltk.tokenize.word_tokenize(s)) for s in all_sentences]
    avg_sentence_length = sum(sentence_lengths) / total_sentences if total_sentences > 0 else 0.0
    
    # Avg. Word Length
    avg_word_length = sum(len(word) for word in all_words) / total_words if total_words > 0 else 0.0

    # Top n Words
    word_freq = Counter(filtered_words_nonstop_nonnum)
    top_n_words = word_freq.most_common(top_n)

    # Top n Abbreviations
    abbreviation_freq = Counter(abbreviations_found)
    top_n_abbreviations = abbreviation_freq.most_common(top_n)
    
    # Negation Cue Frequency (as a rate per 1,000 words)
    negation_cue_rate = (negation_cue_count / total_words) * 1000 if total_words > 0 else 0.0

    # Misspelling Rate (as a percentage)
    # Note: This is an estimation. Some "misspelled" words might be proper nouns or jargon.
    misspelling_rate = (len(misspelled_words) / len([w for w in all_words if w.isalpha()])) * 100 if len([w for w in all_words if w.isalpha()]) > 0 else 0.0
    
    # Lexical Diversity (Type-Token Ratio)
    # Ratio of unique words to total words. Higher is more diverse.
    lexical_diversity = total_unique_words / len(filtered_words_nonstop_nonnum) if len(filtered_words_nonstop_nonnum) > 0 else 0.0

    # Prepare results for output
    results = {
        "Total Documents": total_documents,
        "Total Word Count": total_words,
        "Total Unique Words": total_unique_words,
        "Total Sentences": total_sentences,
        "Avg. Sentence Length (words)": avg_sentence_length,
        "Avg. Word Length (characters)": avg_word_length,
        "Total Non-stop Words Count": total_words_nonstop,
        "Total Unique Non-stop Words": total_unique_words_nonstop,
        "Total Non-stop/Non-num Words Count": total_words_nonstop_nonnum,
        "Total Unique Non-stop/Non-num Words": total_unique_words_nonstop_nonnum,
        f"Top {top_n} Words": top_n_words,
        f"Top {top_n} Abbreviations": top_n_abbreviations,
        "Negation Cue Frequency (per 1k words)": negation_cue_rate,
        "Misspelling Rate (%)": misspelling_rate,
        "Lexical Diversity": lexical_diversity,
        "All Words": all_words,
        "Non-stop Words": filtered_words_nonstop,
        "Non-stop/Non-num Words": filtered_words_nonstop_nonnum,
        "Misspelled Words": misspelled_words
    }
    
    return results


def plot_corpus_word_comparison(
    linguistics_results: list[Dict[str, Any]],
    prompt_type: str,
    top_n: int = 15,
    per_number_words: bool = False,
    corpus_names: list[str] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Plots a comparison of word frequencies across corpora.
    
    First extracts the top n most frequent words from the base corpus,
    then plots the frequency of these same words in all corpora.
    
    Args:
        linguistics_results: The list of dictionaries containing linguistic analysis results for each corpus.
        prompt_type: The type of prompt used for the corpora.
        top_n: Number of top words to compare.
        per_number_words: If True, frequencies are normalized per number of words in each corpus.
        corpus_names: Optional list of names for the corpora, used for the legend.
        output_path: Optional path to save the plot; if None, the plot is shown instead.
    """
    # Set default names if not provided
    if corpus_names is None:
        corpus_names = ["Base Corpus"]
        for i in range(len(augmented_corpus)):
            corpus_names.append(f"Augmented Corpus {i+1}")
    
    # Convert list of word lists into list of frequency dictionaries
    corpus_dicts = [Counter(linguistic["Non-stop/Non-num Words"]) for linguistic in linguistics_results]

    # Get the top n most frequent words from the base corpus
    base_corpus = corpus_dicts[0]
    top_n_words = sorted(base_corpus.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [word for word, _ in top_n_words]
    
    # Collect counts for these words from all corpora
    if per_number_words:
        # Normalize counts by total number of words in each corpus
        count_data = [[(corpus.get(word, 0) / sum(corpus.values())) * 1000 for word in words] for corpus in corpus_dicts]
        plot_title = f"Comparison of top {top_n} words per 1,000 words ({prompt_type})"
        filename = f"FIG2_per_1000_words.jpg"
    else:
        count_data = [[corpus.get(word, 0) for word in words] for corpus in corpus_dicts]
        plot_title = f"Comparison of top {top_n} words ({prompt_type})"
        filename = f"FIG2_absolute_counts.jpg"
    
    # Create grouped bar chart
    plt.figure(figsize=(16, 7), dpi=100)
    bar_width = 0.22
    x_positions = np.arange(len(words))
    
    # Plot bars for each corpus
    for i in range(len(count_data)):
        plt.bar(x_positions + i * bar_width,
                count_data[i],
                width=bar_width,
                label=corpus_names[i],
                alpha=0.85)
    
    # Format chart
    plt.xticks(x_positions + bar_width * 1.5, words, rotation=45, ha='right', fontsize=11)
    plt.xlabel("Word", fontsize=12)
    plt.ylabel("Number of occurrences", fontsize=12)
    plt.title(plot_title, pad=20, fontsize=14)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout() # prevents word labels from being cut off
    
    if output_path:
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save the plot
        BIGGER_SIZE = 14
        plt.rc("font", size=BIGGER_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc("axes", labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
        plt.rc("xtick", labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)   # fontsize of the figure title
        plt.savefig(os.path.join(output_path, filename), format="jpg", dpi=300)
    else:
        plt.show()


def plot_word_frequency_distributions(
    linguistic: Dict[str, Any],
    output_path: Optional[str] = None
):
    """
    Plots word frequency distributions for different filtering levels.

    Args:
        linguistic: The dictionary containing linguistic analysis results.
        output_path: Directory to save the plots. If None, plots are not saved.
    """
    # Prepare frequency counters
    frequency_all_words = Counter(linguistic["All Words"])
    frequency_nonstop_words = Counter(linguistic["Non-stop Words"])
    frequency_nonstop_nonnum_words = Counter(linguistic["Non-stop/Non-num Words"])
    frequency_misspelled_words = Counter(linguistic["Misspelled Words"])
    
    # Plotting options
    options = [
        ("(a) All Words", frequency_all_words, "FIG1a.jpg"),
        ("(b) Non-stop Words", frequency_nonstop_words, "FIG1b.jpg"),
        ("(c) Non-stop/Non-num Words", frequency_nonstop_nonnum_words, "FIG1c.jpg"),
        ("(d) Misspelled Words", frequency_misspelled_words, "FIG1d.jpg")
    ]

    for title, freq_counter, filename in options:
        # Create DataFrame for plotting
        df = pd.DataFrame(sorted(freq_counter.items(), key=lambda x: x[1], reverse=True), columns=["Word", "Frequency"])
        fig, ax = plt.subplots()
        ax.hist(list(df["Frequency"]), bins=100)
        ax.set_yscale("log")
        ax.set_xlabel("Occurrences of Words")
        ax.set_ylabel("Frequency (log)")
        ax.set_title(title)

        if output_path:
            # Save the plot
            BIGGER_SIZE = 14
            plt.rc("font", size=BIGGER_SIZE)          # controls default text sizes
            plt.rc("axes", titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc("axes", labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
            plt.rc("xtick", labelsize=BIGGER_SIZE)    # fontsize of the tick labels
            plt.rc("ytick", labelsize=BIGGER_SIZE)    # fontsize of the tick labels
            plt.rc("legend", fontsize=BIGGER_SIZE)    # legend fontsize
            plt.rc("figure", titlesize=BIGGER_SIZE)   # fontsize of the figure title
            plt.savefig(os.path.join(output_path, filename), format="jpg", dpi=300)
        else:
            plt.show()
    
    plt.close()


def print_corpus_analysis_report(
    corpus: Dict[str, Dict],
    top_n: int = 10,
    output_path: Optional[str] = None,
    output_file: Optional[str] = "linguistic_analysis_report"
):
    """
    Prints the linguistic analysis results in a structured, human-readable format.

    Args:
        corpus: A dictionary where keys are document names and values
                contain entity data for each document.
        top_n: Number of top items to display for frequency-based metrics.
        output_path: If provided, the path to save the report.
        output_file: If provided, the name of the file to save the report.
    """
    # Perform linguistic analysis
    linguistic = perform_linguistic_analysis(corpus, top_n=top_n)

    # Generate report
    lines = []

    # Create DataFrame
    total_documents = linguistic["Total Documents"]
    stats_df = pd.DataFrame({
        "Metric": [
            "Total Documents",
            "Total Word Count",
            "Total Unique Words",
            "Total Sentences",
            "Total Non-stop Words Count",
            "Total Unique Non-stop Words",
            "Total Non-stop/Non-num Words Count",
            "Total Unique Non-stop/Non-num Words",
            "Average Sentence Length (words)",
            "Average Word Length (characters)",
            "Lexical Diversity (Type-Token Ratio)",
            "Negation Cue Frequency (per 1k words)",
            "Misspelling Rate (%)"
        ],
        "Absolute": [
            linguistic["Total Documents"],
            f"{linguistic['Total Word Count']:,}",
            f"{linguistic['Total Unique Words']:,}",
            f"{linguistic['Total Sentences']:,}",
            f"{linguistic['Total Non-stop Words Count']:,}",
            f"{linguistic['Total Unique Non-stop Words']:,}",
            f"{linguistic['Total Non-stop/Non-num Words Count']:,}",
            f"{linguistic['Total Unique Non-stop/Non-num Words']:,}",
            f"{linguistic['Avg. Sentence Length (words)']:.2f}",
            f"{linguistic['Avg. Word Length (characters)']:.2f}",
            f"{linguistic['Lexical Diversity']:.4f}",
            f"{linguistic['Negation Cue Frequency (per 1k words)']:.2f}",
            f"{linguistic['Misspelling Rate (%)']:.2f}"
        ],
        "Relative": [
            "-",
            f"{linguistic['Total Word Count']/total_documents:.2f}",
            f"{linguistic['Total Unique Words']/total_documents:.2f}",
            f"{linguistic['Total Sentences']/total_documents:.2f}",
            f"{linguistic['Total Non-stop Words Count']/total_documents:.2f}",
            f"{linguistic['Total Unique Non-stop Words']/total_documents:.2f}",
            f"{linguistic['Total Non-stop/Non-num Words Count']/total_documents:.2f}",
            f"{linguistic['Total Unique Non-stop/Non-num Words']/total_documents:.2f}",
            "-", "-", "-", "-", "-"
        ]
    })

    # Append corpus statistics to report
    lines.append("CORPUS STATISTICS:")
    lines.append("-" * 67)
    lines.append(stats_df.to_markdown(index=False, tablefmt="pipe", colalign=["left", "right", "right"]))
    lines.append("-" * 67)
    
    lines.append(f"Top {top_n} Most Frequent Words (excluding stopwords):")
    for word, count in linguistic[f"Top {top_n} Words"]:
        lines.append(f"  - '{word}': {count} times")
    lines.append("-" * 50)
    lines.append(f"Top {top_n} Most Frequent Abbreviations:")
    if linguistic[f"Top {top_n} Abbreviations"]:
        for abbr, count in linguistic[f"Top {top_n} Abbreviations"]:
            lines.append(f"  - '{abbr}': {count} times")
    else:
        lines.append("  - None found.")
    lines.append("-" * 50)

    # Combine lines into final report
    report = "\n".join(lines)
    logger.info(report)
    
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Save report to txt file
        output_file = os.path.join(output_path, output_file)
        with open(output_file + ".txt", "w", encoding="utf-8") as f:
            f.write(report)
        stats_df.to_csv(output_file + ".csv", index=False, sep=";")
        logger.info(f"Report of validation saved to {output_file}")

    # Plot word frequency distributions
    plot_word_frequency_distributions(linguistic, output_path)


def print_corpus_word_comparison(
    data_path: str,
    prompt_type: str,
    models: list[str],
    top_n: int = 15,
    output_path: Optional[str] = None
) -> None:
    """
    Plots a comparison of word frequencies across corpora.
    
    First extracts the top n most frequent words from the base corpus,
    then plots the frequency of these same words in all corpora.
    
    Args:
        data_path: Base path for the corpora.
        corpora_paths: List of paths to the corpora.
        top_n: Number of top words to compare.
        corpus_names: Optional list of names for the corpora, used for the legend.
        output_path: Optional path to save the plot; if None, the plot is shown instead.
    """
    # Define paths and names for corpora to compare
    corpora_paths = ["raw/train"]
    corpus_names = ["Original"]

    for model in models:
        # Add the augmented corpus path for this model and prompt to the list of corpora paths to compare
        augmented_corpus_path = f"augmented/{prompt_type}/{model}"
        corpora_paths.append(augmented_corpus_path)
        corpus_names.append(f"{model}")
    
    # Initialize corpus dictionaries list
    linguistics_results = []

    for corpus_path in corpora_paths:
        # Load the complete corpus for each path and append to the list
        corpus = load_complete_corpus(data_path=os.path.join(data_path, corpus_path))
        
        # Perform linguistic analysis
        linguistic = perform_linguistic_analysis(corpus, top_n=top_n)
        linguistics_results.append(linguistic)

    # Plot the comparison of word frequencies across corpora
    for per_number in [False, True]:
        plot_corpus_word_comparison(
            linguistics_results,
            top_n=top_n,
            prompt_type=prompt_type,
            per_number_words=per_number,
            corpus_names=corpus_names,
            output_path=output_path
        )
