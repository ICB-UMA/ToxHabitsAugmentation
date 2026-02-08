#!/usr/bin/env python3
"""
NER Datasets

This module defines the TransformersNERDataset class, which is a PyTorch Dataset
for Named Entity Recognition (NER) tasks using Hugging Face tokenizers. It handles
the alignment of word-level IOB2 tags with subword tokens, ensuring that only the
first subtoken of each word receives the label id, while the rest are assigned
a padding label id to be ignored during loss calculation.
"""

# --- Local Imports ---
from ..imports import (
    pd,
    torch,
    List,
    Dict,
    Any
)


class TransformersNERDataset(torch.utils.data.Dataset):
    """
    Dataset for NER tasks using a HuggingFace tokenizer.

    Aligns word-level IOB2 tags with subword tokens. Only the first subtoken of each
    word receives the label id; the rest receive `label_pad_id` so they are ignored
    by the loss.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        max_len: int,
        label_pad_id: int = -100,
        tags_col: str = "iob_tags",
    ):
        """
        Initializes the dataset with the given data, tokenizer, and configuration.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset with words and tags.
            tokenizer (Any): Hugging Face tokenizer for tokenizing the input words.
            max_len (int): Maximum sequence length for tokenization.
            label_pad_id (int, optional): Label ID used for padding tokens. Defaults to -100.
            tags_col (str, optional): Column name in the DataFrame containing the IOB2 tags. Defaults to "iob_tags".
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.label_pad_id = int(label_pad_id)
        self.tags_col = tags_col

    def __len__(
        self
    ) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(
        self,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves the sample at the specified index, tokenizes the input words, and aligns the labels.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized input and aligned labels.
        """
        row = self.data.iloc[idx]
        words: List[str] = row["words"]

        tag_ids: List[str] = row[self.tags_col]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(0)
        aligned_labels: List[int] = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(self.label_pad_id)
            elif word_idx != prev_word_idx:
                aligned_labels.append(tag_ids[word_idx])
            else:
                aligned_labels.append(self.label_pad_id)
            prev_word_idx = word_idx

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return item
