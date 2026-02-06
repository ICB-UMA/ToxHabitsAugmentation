#!/usr/bin/env python3
"""
NER Preprocessing Utilities

This module provides functions to preprocess text data for Named Entity Recognition (NER) tasks.
It includes functions to extract sentence positions, assign IOB tags, tokenize and align data,
and build tagged DataFrames from raw text and annotations.
"""

# --- Third-Party Imports ---
import unicodedata
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize

# --- Local Imports ---
from .logging import logger
from ..imports import (
    os,
    pd,
    re,
    torch,
    Dict,
    List,
    Tuple,
    Optional
)


def get_sentence_positions(text: str) -> List[Tuple[int, int]]:
    """
    Extract positions of sentences from a given text.

    Args:
        text (str): The input text to be processed.
    
    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple contains the start and 
                               end character positions of a sentence in the text. 
    """
    sentences = sent_tokenize(text)
    positions = []
    start = 0
    
    for sentence in sentences:
        start = text.find(sentence, start)
        end = start + len(sentence)
        positions.append((start, end))
        start = end

    return positions


def assign_iob_tags(
    row: pd.Series,
    df_ann: pd.DataFrame,
    label2id: Dict[str, int],
) -> List[int]:
    """
    Build token-level IOB2 tag ids for a sentence/fragment.

    For each word start offset, find the gold annotation span covering it.
    If the word start == ann_start -> B-<LABEL>, else -> I-<LABEL>.
    Otherwise -> O.

    Args:
        row (pd.Series): A row from the DataFrame containing 'filename' and 'start_end' (word-level offsets).
        df_ann (pd.DataFrame): DataFrame containing the annotations with columns 'filename',
                               'start_span', 'end_span', and 'label'.
        label2id (Dict[str, int]): A dictionary mapping IOB tags to their corresponding integer IDs.

    Returns:
        List[int]: A list of label ids aligned with row['start_end'] (word-level).
    """
    filename = row["filename"]
    word_ranges = row["start_end"]

    # Keep only this document's annotations
    doc_ann = df_ann[df_ann["filename"] == filename][["start_span", "end_span", "label"]].values

    iob_ids: List[int] = []

    for start, _ in word_ranges:
        tag = "O"

        for ann_start, ann_end, ann_label in doc_ann:
            if int(ann_start) <= int(start) < int(ann_end):
                prefix = "B" if int(start) == int(ann_start) else "I"
                tag = f"{prefix}-{ann_label}"
                break

        iob_ids.append(label2id[tag])

    return iob_ids


def tokenize_and_align_from_df(df, tokenizer, label2id, max_len):
    all_encodings = []
    all_labels = []

    for _, row in df.iterrows():
        words = row['words']
        tags = row['iob_tags']
        tag_ids = [label2id[tag] for tag in tags]

        tokenized_inputs = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

        word_ids = tokenized_inputs.word_ids(batch_index=0)
        aligned_labels = [-100 if idx is None else tag_ids[idx] for idx in word_ids]

        all_encodings.append(tokenized_inputs)
        all_labels.append(aligned_labels)

    input_ids = torch.cat([enc['input_ids'] for enc in all_encodings])
    attention_mask = torch.cat([enc['attention_mask'] for enc in all_encodings])
    labels = torch.cat([torch.tensor(lbl).unsqueeze(0) for lbl in all_labels])

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

def process_files_in_directory(directory: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Process all .txt files in a directory to extract sentence positions.

    Parameters:
    -----------
    directory : str
        The path to the directory containing .txt files.

    Returns:
    --------
    Dict[str, List[Tuple[int, int]]]
        A dictionary where keys are file names (without extensions) and values are lists of sentence positions.
    """
    sentence_positions_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                sentence_positions = get_sentence_positions(text)
                file_key = os.path.splitext(filename)[0]
                sentence_positions_dict[file_key] = sentence_positions
                
    return sentence_positions_dict

def load_texts_to_dict(txt_dir: str) -> Dict[str, str]:
    """
    Load all .txt files from a directory into a dictionary.

    Parameters:
    -----------
    txt_dir : str
        The path to the directory containing .txt files.

    Returns:
    --------
    Dict[str, str]
        A dictionary where keys are file names (without extensions) and values are the raw text content.
    """
    texts_dict = {}

    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(txt_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                texts_dict[filename[:-4]] = raw_text
    
    return texts_dict

def is_punctuation(ch: str) -> bool:
    """
    Check if a character is a punctuation mark.

    Parameters:
    -----------
    ch : str
        The character to be checked.

    Returns:
    --------
    bool
        True if the character is a punctuation mark, otherwise False.
    """
    return unicodedata.category(ch).startswith('P')

def is_cjk_character(ch: str) -> bool:
    """
    Check if a character is a CJK (Chinese, Japanese, Korean) character.

    Parameters:
    -----------
    ch : str
        The character to be checked.

    Returns:
    --------
    bool
        True if the character is CJK, otherwise False.
    """
    return any(
        [
            '\u4e00' <= ch <= '\u9fff',
            '\u3400' <= ch <= '\u4dbf',
            '\u20000' <= ch <= '\u2a6df',
            '\u2a700' <= ch <= '\u2b73f',
            '\u2b740' <= ch <= '\u2b81f',
            '\u2b820' <= ch <= '\u2ceaf',
            '\uf900' <= ch <= '\ufaff',
            '\u2f800' <= ch <= '\u2fa1f'
        ]
    )

def is_space(ch: str) -> bool:
    """
    Check if a character is a space character.

    Parameters:
    -----------
    ch : str
        The character to be checked.

    Returns:
    --------
    bool
        True if the character is a space, otherwise False.
    """
    return ch in ' \t\n\r\f\v'

def is_control(ch: str) -> bool:
    """
    Check if a character is a control character.

    Parameters:
    -----------
    ch : str
        The character to be checked.

    Returns:
    --------
    bool
        True if the character is a control character, otherwise False.
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def word_start_end(text: str, start_i: int = 0, cased: bool = True) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Extract word-level tokens and their start-end positions from a text.

    Parameters:
    -----------
    text : str
        The input text to be tokenized.
    start_i : int, optional (default=0)
        The starting index position.
    cased : bool, optional (default=True)
        Whether to preserve case during tokenization.

    Returns:
    --------
    Tuple[List[str], List[Tuple[int, int]]]
        A tuple containing a list of words and a list of corresponding start-end positions.
    """
    if not cased:
        text = ''.join(ch for ch in unicodedata.normalize('NFD', text) if unicodedata.category(ch) != 'Mn').lower()
    
    spaced, arr_start = '', []
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            arr_start.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not (ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            arr_start.append(start_i)
        start_i += 1
        
    arr_word, arr_start_end = [], []
    i = 0
    for word in spaced.strip().split():
        arr_word.append(word)
        j = i + len(word)
        arr_start_end.append((arr_start[i], arr_start[j - 1] + 1))
        i = j  
    return arr_word, arr_start_end


def text2sentencesplit(texts_dict: Dict[str, str], ss_dict: Dict[str, List[Tuple[int, int]]], tokenizer, sub_tok_max_seq_len: int, cased: bool = True) -> pd.DataFrame:
    """
    Splits documents into sentences based on token length constraints and word offsets.

    Parameters:
    -----------
    texts_dict : Dict[str, str]
        A dictionary where keys are document identifiers and values are the document texts.
    ss_dict : Dict[str, List[Tuple[int, int]]]
        A dictionary where keys are document identifiers and values are lists of (start, end) tuples
        representing the start and end character positions of sentences in the document.
    tokenizer : object
        A tokenizer object with a `tokenize()` method to convert words into subword tokens.
    sub_tok_max_seq_len : int
        The maximum number of tokens allowed in each sentence fragment.
    cased : bool, optional (default=True)
        A flag indicating whether the tokenization should preserve the case of the text.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to a sentence fragment that fits within the token limit.
        The DataFrame contains the following columns:
        - 'filename': The identifier of the document.
        - 'words': The list of words in the fragment.
        - 'start_end': A list of tuples representing the start and end character positions of each word.
        - 'num_tokens': The total number of tokens in the fragment.
    """
    data = []
    for doc, doc_text in texts_dict.items():
        doc_ss = ss_dict[doc]
        
        for ss_start, ss_end in doc_ss:
            ss_arr_word, ss_arr_word_start_end = word_start_end(text=doc_text[ss_start:ss_end], start_i=ss_start, cased=cased)

            current_tokens = 0
            current_words = []
            current_word_start_end = []
            
            for idx, (word, (start, end)) in enumerate(zip(ss_arr_word, ss_arr_word_start_end)):
                word_tokens = tokenizer.tokenize(word)
                num_word_tokens = len(word_tokens)

                if current_tokens + num_word_tokens > sub_tok_max_seq_len:
                    data.append({
                        'filename': doc,
                        'words': current_words,
                        'start_end': current_word_start_end,
                        'num_tokens': current_tokens
                    })

                    current_tokens = 0
                    current_words = []
                    current_word_start_end = []
                
                current_words.append(word)
                current_word_start_end.append((start, end))
                current_tokens += num_word_tokens
            
            if current_words:
                data.append({
                    'filename': doc,
                    'words': current_words,
                    'start_end': current_word_start_end,
                    'num_tokens': current_tokens
                })

    return pd.DataFrame(data)

def greedy_strategy(df: pd.DataFrame, sub_tok_max_seq_len: int) -> pd.DataFrame:
    """
    Greedily accumulates sentences into fragments until the token limit is reached.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing sentence information with the following columns:
        - 'filename': The identifier of the document.
        - 'words': List of words in the sentence.
        - 'start_end': List of tuples with start and end character positions of each word.
        - 'num_tokens': The number of tokens in the sentence.
    sub_tok_max_seq_len : int
        The maximum number of tokens allowed in each fragment.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to a fragment of sentences that together do not exceed the token limit.
        The DataFrame contains the following columns:
        - 'filename': The identifier of the document.
        - 'words': The list of words in the fragment.
        - 'start_end': A list of tuples representing the start and end character positions of each word.
        - 'num_tokens': The total number of tokens in the fragment.
    """
    greedy_data = []
    i = 0

    while i < len(df):
        row = df.iloc[i]
        current_filename = row['filename']
        current_words = row['words']
        current_word_start_end = row['start_end']
        current_num_tokens = row['num_tokens']

        j = i + 1
        while j < len(df):
            next_row = df.iloc[j]

            if next_row['filename'] != current_filename:
                break

            if current_num_tokens + next_row['num_tokens'] <= sub_tok_max_seq_len:
                current_words.extend(next_row['words'])
                current_word_start_end.extend(next_row['start_end'])
                current_num_tokens += next_row['num_tokens']
            else:
                break

            j += 1

        greedy_data.append({
            'filename': current_filename,
            'words': current_words,
            'start_end': current_word_start_end,
            'num_tokens': current_num_tokens
        })

        i = j

    return pd.DataFrame(greedy_data)



def extract_entities_to_df(df: pd.DataFrame, texts_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Extracts named entities from a DataFrame containing predicted labels and word offsets, and returns a DataFrame of entities.

    Parameters:
    ----------- 
    df : pd.DataFrame
        A DataFrame that contains the following columns:
        - 'filename': The identifier of the document.
        - 'labels': A list of predicted labels for each word or token in the document.
        - 'start_end': A list of tuples representing the start and end character offsets for each word.
    texts_dict : Dict[str, str]
        A dictionary where the keys are document identifiers and the values are the corresponding document texts.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to an extracted named entity. The DataFrame contains the following columns:
        - 'filename': The identifier of the document.
        - 'label': The entity type (e.g., "PERSON", "ORG", etc.).
        - 'start_span': The start character offset of the entity.
        - 'end_span': The end character offset of the entity.
        - 'text': The extracted entity text from the document.

    Notes:
    ------
    - The function looks for entity labels that start with "B-" to mark the beginning of a new entity and "I-" to continue the same entity.
    - When an entity ends (upon encountering an "O" label or a new "B-" label), the entity is closed and added to the result.
    """
    entities = []
    for _, row in df.iterrows():
        text = texts_dict[row["filename"]]
        current_entity = None
        start_idx = None

        for label, (start, end) in zip(row["labels"], row["start_end"]):

            if label.startswith("B-"):
                if current_entity:
                    entity_text = text[start_idx[0]:start_idx[1]]
                    entities.append({
                        'filename': row['filename'],
                        'label': current_entity,
                        'start_span': start_idx[0],
                        'end_span': start_idx[1],
                        'text': entity_text
                    })
                current_entity = label[2:]
                start_idx = (start, end)

            elif label.startswith("I-") and current_entity:
                start_idx = (start_idx[0], end)

            elif label == "O" or label.startswith("B-"):
                if current_entity:
                    entity_text = text[start_idx[0]:start_idx[1]]
                    entities.append({
                        'filename': row['filename'],
                        'label': current_entity,
                        'start_span': start_idx[0],
                        'end_span': start_idx[1],
                        'text': entity_text
                    })
                current_entity = None  

        if current_entity:
            entity_text = text[start_idx[0]:start_idx[1]]
            entities.append({
                'filename': row['filename'],
                'label': current_entity,
                'start_span': start_idx[0],
                'end_span': start_idx[1],
                'text': entity_text
            })

    df_entities = pd.DataFrame(entities, columns=["filename", "label", "start_span", "end_span", "text"])

    return df_entities



def check_overlapped(df: pd.DataFrame) -> None:
    """
    Checks for overlapping annotation spans in the DataFrame.

    Logs whether overlaps exist and shows up to two overlapping examples.
    """
    overlapped_pairs = []
    for filename in df['filename'].unique():
        doc_df = df[df['filename'] == filename].sort_values(by='start_span')
        spans = doc_df[['start_span', 'end_span']].values
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if spans[i][1] > spans[j][0]:  # overlap detected
                    overlapped_pairs.append((doc_df.iloc[i], doc_df.iloc[j]))
    if overlapped_pairs:
        logger.info("Overlapping annotations were found in the dataset.")
        for a, b in overlapped_pairs[:2]:
            logger.info(f"\n--- Annotation 1 ---\n{a.to_dict()}\n--- Annotation 2 ---\n{b.to_dict()}")
    else:
        logger.info("No overlapping annotations found.")

def eliminate_overlapped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminates overlapping annotations by keeping the largest span
    and removing any smaller spans fully contained within it.

    Returns a cleaned DataFrame.
    """
    cleaned_rows = []
    for filename in df['filename'].unique():
        doc_df = df[df['filename'] == filename].copy()
        doc_df['length'] = doc_df['end_span'] - doc_df['start_span']
        doc_df = doc_df.sort_values(by='length', ascending=False)

        keep_spans = []
        for _, row in doc_df.iterrows():
            span = (row['start_span'], row['end_span'])
            if not any((span[0] < kept[1] and span[1] > kept[0]) for kept in keep_spans):
                keep_spans.append(span)
                cleaned_rows.append(row)
    return pd.DataFrame(cleaned_rows).drop(columns='length')


def build_tagged_df(
    data_path: str,
    tokenizer,
    sub_tok_max: int,
    label2id: Dict[str,int],
    ann_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build tagged dataframe (IOB) from a directory containing txt + ann.
    """
    texts = load_texts_to_dict(data_path)
    ss_dict = process_files_in_directory(data_path)

    df = text2sentencesplit(texts, ss_dict, tokenizer, sub_tok_max)
    df = greedy_strategy(df, sub_tok_max)

    df["iob_tags"] = df.apply(lambda r: assign_iob_tags(r, ann_df, label2id), axis=1)


    return df

def create_label_dicts(tags: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    unique_tags = sorted(set(tags))

    id2label = {0: 'O'}
    label2id = {'O': 0}

    idx = 1
    for tag in unique_tags:
        for prefix in ['B', 'I']:
            label = f"{prefix}-{tag}"
            id2label[idx] = label
            label2id[label] = idx
            idx += 1

    return id2label, label2id


def extract_entities_with_scores(
    df: pd.DataFrame,
    texts_dict: Dict[str, str],
    labels_col: str = "labels",
    scores_col: str = "label_scores",
    offsets_col: str = "start_end",
    filename_col: str = "filename",
) -> pd.DataFrame:
    """
    Extract IOB2 entities from a dataframe with word-level labels and per-word scores.

    Expected columns per row:
      - filename_col: str
      - offsets_col: List[Tuple[int,int]]  (word start/end offsets)
      - labels_col: List[str]             (IOB2 labels per word, e.g. "B-DRUG", "I-DRUG", "O")
      - scores_col: List[float]           (score per word)

    Returns a flat dataframe with:
      filename, label, start_span, end_span, text, score
    """
    entities: List[Dict] = []

    for _, row in df.iterrows():
        filename = str(row[filename_col])
        text = texts_dict[filename]

        labels = row[labels_col]
        scores = row[scores_col]
        offsets = row[offsets_col]

        current_entity: Optional[str] = None
        span_start: Optional[int] = None
        span_end: Optional[int] = None
        score_accum: List[float] = []

        for lab, sc, (start, end) in zip(labels, scores, offsets):
            if lab.startswith("B-"):
                # close previous entity if any
                if current_entity is not None:
                    entity_text = text[span_start:span_end]
                    entities.append(
                        {
                            "filename": filename,
                            "label": current_entity,
                            "start_span": span_start,
                            "end_span": span_end,
                            "text": entity_text,
                            "score": float(sum(score_accum) / max(len(score_accum), 1)),
                        }
                    )

                # start new entity
                current_entity = lab[2:]
                span_start = int(start)
                span_end = int(end)
                score_accum = [float(sc)]

            elif lab.startswith("I-") and current_entity is not None:
                # continue entity
                span_end = int(end)
                score_accum.append(float(sc))

            else:
                # lab == "O" OR invalid I- without current entity -> close if open
                if current_entity is not None:
                    entity_text = text[span_start:span_end]
                    entities.append(
                        {
                            "filename": filename,
                            "label": current_entity,
                            "start_span": span_start,
                            "end_span": span_end,
                            "text": entity_text,
                            "score": float(sum(score_accum) / max(len(score_accum), 1)),
                        }
                    )
                current_entity = None
                span_start = None
                span_end = None
                score_accum = []

        # close last entity
        if current_entity is not None:
            entity_text = text[span_start:span_end]
            entities.append(
                {
                    "filename": filename,
                    "label": current_entity,
                    "start_span": span_start,
                    "end_span": span_end,
                    "text": entity_text,
                    "score": float(sum(score_accum) / max(len(score_accum), 1)),
                }
            )

    return pd.DataFrame(
        entities,
        columns=["filename", "label", "start_span", "end_span", "text", "score"],
    )


def _word_level_from_token_logits(
    token_logits: torch.Tensor,
    word_ids: List[Optional[int]],
) -> Tuple[List[int], List[float]]:
    """
    Collapse token-level logits to word-level predictions by taking the FIRST token of each word.

    Args:
        token_logits: [seq_len, num_labels] tensor (on CPU)
        word_ids: list length seq_len with word index or None

    Returns:
        word_pred_ids: list[int] length n_words
        word_scores:   list[float] length n_words, score = max softmax prob for the chosen label
    """
    probs = torch.softmax(token_logits, dim=-1)  # [L, C]

    first_tok_idx_per_word: Dict[int, int] = {}
    for t_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if w_id not in first_tok_idx_per_word:
            first_tok_idx_per_word[w_id] = t_idx

    if not first_tok_idx_per_word:
        return [], []

    n_words = max(first_tok_idx_per_word.keys()) + 1
    word_pred_ids: List[int] = []
    word_scores: List[float] = []

    for w in range(n_words):
        t_idx = first_tok_idx_per_word.get(w, None)
        if t_idx is None:
            # shouldn't happen, but keep alignment stable
            word_pred_ids.append(0)
            word_scores.append(0.0)
            continue

        p = probs[t_idx]  # [C]
        pred_id = int(torch.argmax(p).item())
        score = float(torch.max(p).item())

        word_pred_ids.append(pred_id)
        word_scores.append(score)

    return word_pred_ids, word_scores


def predict_and_extract_entities(
    df: pd.DataFrame,
    model: torch.nn.Module,
    id2label: Dict[int, str],
    texts_dict: Dict[str, str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Predict word-level IOB2 labels for a dataframe with columns:
      - 'filename'
      - 'words' (List[str])
      - 'start_end' (List[Tuple[int,int]])

    Then extract entities with mean score per entity.

    Notes:
      - Uses tokenizer(..., is_split_into_words=True) to get word_ids reliably.
      - Uses FIRST subtoken per word to decide label+score.
    """
    tokenizer = model.get_tokenizer()
    sub_tok_max_seq_len = model.get_max_len(max_len)

    df = df.copy()
    all_labels: List[List[str]] = []
    all_label_scores: List[List[float]] = []

    model.eval()
    model.to(device)

    # simple batching over rows (not torch Dataset to keep it plug-and-play)
    rows = df.to_dict(orient="records")

    for i in tqdm(range(0, len(rows), batch_size), desc="Evaluating (span reconstruction)", leave=False):
        batch_rows = rows[i : i + batch_size]
        batch_words = [r["words"] for r in batch_rows]

        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=sub_tok_max_seq_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, L, C]

        logits = logits.detach().cpu()

        for b_idx, r in enumerate(batch_rows):
            word_ids = enc.word_ids(batch_index=b_idx)  # List[Optional[int]]

            word_pred_ids, word_scores = _word_level_from_token_logits(
                token_logits=logits[b_idx],
                word_ids=word_ids,
            )

            # map ids -> labels
            labels = [id2label[int(x)] for x in word_pred_ids]

            # ensure alignment with start_end length
            n_words_expected = len(r["start_end"])
            if len(labels) < n_words_expected:
                labels = labels + ["O"] * (n_words_expected - len(labels))
                word_scores = word_scores + [0.0] * (n_words_expected - len(word_scores))
            else:
                labels = labels[:n_words_expected]
                word_scores = word_scores[:n_words_expected]

            all_labels.append(labels)
            all_label_scores.append(word_scores)

    df["labels"] = all_labels
    df["label_scores"] = all_label_scores

    return extract_entities_with_scores(
        df=df,
        texts_dict=texts_dict,
        labels_col="labels",
        scores_col="label_scores",
        offsets_col="start_end",
        filename_col="filename",
    )


def load_ann_to_dataframe(
    ann_dir: str
) -> pd.DataFrame:
    """
    Parse all .ann files in a given directory (BRAT-style annotations) and extract
    entity annotations together with their semantic labels and associated concept
    codes (CXXXXXXX) stored in annotator notes.

    The function supports discontinuous spans by merging all span fragments into
    a single continuous span (min start, max end).

    Parameters
    ----------
    ann_dir : str
        Path to the directory containing .ann files.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - filename   : document identifier (without extension)
        - start_span : starting character offset of the entity
        - end_span   : ending character offset of the entity
        - text       : entity surface form
        - label      : entity category (e.g., DISEASE, SYMPTOM, etc.)
        - codes      : list of associated concept codes (e.g., ["C1234567"])
    """
    data = []

    for filename in os.listdir(ann_dir):
        if not filename.endswith(".ann"):
            continue

        filepath = os.path.join(ann_dir, filename)
        file_id = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        entities = {}
        notes = {}

        for line in lines:
            line = line.strip()

            if line.startswith("T"):
                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                tid = parts[0]
                span_info = parts[1]
                text = parts[2]

                span_parts = span_info.split()
                label = span_parts[0]

                spans_raw = " ".join(span_parts[1:])
                nums = re.findall(r"\d+", spans_raw)

                starts = [int(nums[i]) for i in range(0, len(nums), 2)]
                ends = [int(nums[i]) for i in range(1, len(nums), 2)]

                start = min(starts)
                end = max(ends)

                entities[tid] = {
                    "start": start,
                    "end": end,
                    "text": text,
                    "label": label,
                }

            elif line.startswith("#"):
                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                _, ann_info, note_text = parts
                match = re.search(r"AnnotatorNotes (T\d+)", ann_info)

                if match:
                    tid = match.group(1)
                    codes = re.findall(r"C\d{7}", note_text)

                    if tid in notes:
                        notes[tid].extend(codes)
                    else:
                        notes[tid] = codes

        for tid, entity in entities.items():
            code_list = notes.get(tid, [])

            data.append(
                {
                    "filename": file_id,
                    "start_span": entity["start"],
                    "end_span": entity["end"],
                    "text": entity["text"],
                    "label": entity["label"],
                    "codes": code_list,
                }
            )

    return pd.DataFrame(data)
