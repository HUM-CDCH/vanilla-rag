"""Token-based chunking.

Documents are segmented into fixed windows of `chunk_size` tokens, counted
with the embedding model's own tokenizer so chunk boundaries and the 5K
context budget use the same unit.
"""

from transformers import PreTrainedTokenizerBase


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 512,
    overlap: int = 0,
) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` tokens.

    Chunks are sliced from the original string via token offset mappings —
    decoding token ids back to text would lowercase/normalize it with
    uncased tokenizers like MiniLM's.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + chunk_size]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(offsets):
            break
    return chunks


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
