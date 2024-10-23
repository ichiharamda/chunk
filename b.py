from __future__ import annotations

import re
from bisect import bisect_left
from typing import Callable, Sequence
from functools import cache
from itertools import accumulate
from contextlib import suppress

import tiktoken
from mpire import WorkerPool
from tqdm import tqdm

# 定数：意味のある句読点や文字列区切り
_NON_WHITESPACE_SEMANTIC_SPLITTERS = (
    '.', '?', '!', '*', # 文の区切り
    ';', ',', '(', ')', '[', ']', "“", "”", '‘', '’', "'", '"', '`', # 節の区切り
    ':', '—', '…', '/', '\\', '–', '&', '-', # 単語の結合
)

# テキストをセマンティクスに基づいて分割
def _split_text(text: str) -> tuple[str, bool, list[str]]:
    splitter_is_whitespace = True

    if '\n' in text or '\r' in text:
        splitter = max(re.findall(r'[\r\n]+', text))
    elif '\t' in text:
        splitter = max(re.findall(r'\t+', text))
    elif re.search(r'\s', text):
        splitter = max(re.findall(r'\s+', text))
    else:
        for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS:
            if splitter in text:
                splitter_is_whitespace = False
                break
        else:
            return '', splitter_is_whitespace, list(text)
    return splitter, splitter_is_whitespace, text.split(splitter)

# 分割したテキストをチャンクとして結合
def merge_splits(splits: list[str], chunk_size: int, splitter: str, token_counter: Callable) -> tuple[int, str]:
    average = 0.2
    low = 0
    high = len(splits) + 1
    cumulative_lengths = list(accumulate([len(split) for split in splits], initial=0))
    cumulative_lengths.append(cumulative_lengths[-1])

    while low < high:
        i = bisect_left(cumulative_lengths[low : high + 1], chunk_size * average)
        midpoint = min(i + low, high - 1)
        tokens = token_counter(splitter.join(splits[:midpoint]))
        average = cumulative_lengths[midpoint] / tokens if cumulative_lengths[midpoint] and tokens > 0 else average
        if tokens > chunk_size:
            high = midpoint
        else:
            low = midpoint + 1

    return low - 1, splitter.join(splits[:low - 1])

# テキストをチャンクサイズに基づいて分割
def chunk(
    text: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
    memoize: bool = True,
    _recursion_depth: int = 0,
    _reattach_whitespace_splitters: bool = False,
) -> list[str]:
    splitter, splitter_is_whitespace, splits = _split_text(text)
    if _reattach_whitespace_splitters: splitter_is_whitespace = False
    
    chunks = []
    skips = set()

    for i, split in enumerate(splits):
        if i in skips:
            continue
        
        if token_counter(split) > chunk_size:
            chunks.extend(chunk(split, chunk_size, token_counter = token_counter, memoize = memoize, _recursion_depth = _recursion_depth + 1, _reattach_whitespace_splitters = _reattach_whitespace_splitters))
        else:
            final_split_in_chunk_i, new_chunk = merge_splits(splits[i:], chunk_size, splitter, token_counter)
            skips.update(range(i + 1, i + final_split_in_chunk_i))
            chunks.append(new_chunk)

        if not splitter_is_whitespace and not (i == len(splits) - 1 or all(j in skips for j in range(i + 1, len(splits)))):
            if token_counter(last_chunk_with_splitter := chunks[-1] + splitter) <= chunk_size:
                chunks[-1] = last_chunk_with_splitter
            else:
                chunks.append(splitter)
    
    if not _recursion_depth:
        chunks = list(filter(None, chunks))
    
    return chunks

# Chunkerクラス
class Chunker:    
    def __init__(self, chunk_size: int, token_counter: Callable[[str], int]) -> None:
        self.chunk_size = chunk_size
        self.token_counter = token_counter
    
    def chunk(self, text: str) -> list[str]:
        return chunk(text, self.chunk_size, self.token_counter, memoize=False)
    
    def __call__(
        self,
        text_or_texts: str | Sequence[str],
        processes: int = 1,
        progress: bool = False,
    ) -> list[str] | list[list[str]]:
        if isinstance(text_or_texts, str):
            return self.chunk(text_or_texts)
        
        if progress and processes == 1:
            text_or_texts = tqdm(text_or_texts)
        
        if processes == 1:
            return [self.chunk(text) for text in text_or_texts]
        
        with WorkerPool(processes, use_dill=True) as pool:
            return pool.map(self.chunk, text_or_texts, progress_bar=progress)

# トークナイザーの生成
def chunkerify(
    tokenizer_or_token_counter: str,
    chunk_size: int | None = None,
) -> Chunker:
    tokenizer = tiktoken.encoding_for_model(tokenizer_or_token_counter)
    token_counter = lambda text: len(tokenizer.encode(text))

    if chunk_size is None:
        chunk_size = tokenizer.model_max_length - len(tokenizer.encode(''))
    
    return Chunker(chunk_size, token_counter)

# メイン部分
if __name__ == "__main__":
    # トークナイザーの生成（GPT-3.5-turboの例）
    chunker = chunkerify("gpt-3.5-turbo", chunk_size=1000)

    # 長いテキストの例
    long_text = """
    In the year 2024, technological advancements have accelerated at an unprecedented pace. 
    AI has seamlessly integrated into everyday life, revolutionizing industries, healthcare, and education. 
    Autonomous vehicles have become a common sight on roads, drastically reducing accidents and traffic congestion. 
    The healthcare sector has been transformed by AI-driven diagnostics, enabling early detection of diseases with unparalleled accuracy. 
    Education has evolved with personalized learning systems, allowing students to progress at their own pace. 
    However, these advancements have also raised ethical questions about privacy, job displacement, and the potential for misuse of technology.
    """

    # テキストのチャンク化
    chunks = chunker(long_text)

    # チャンクの表示
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
