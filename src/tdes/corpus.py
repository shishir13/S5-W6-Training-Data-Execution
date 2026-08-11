"""Synthetic corpus generation — one document per lane, varied content."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    doc_id: str
    lane: str
    text: str
    is_eval: bool = False


_ENGLISH_WEB: List[str] = [
    "The history of computing spans decades of innovation. "
    "Early pioneers built mechanical calculators, then vacuum tube machines, "
    "then transistor-based systems. Each generation reduced size and cost while "
    "increasing speed and reliability. Today's processors contain billions of transistors "
    "and execute trillions of operations per second.",

    "Climate change represents one of the most significant challenges facing humanity. "
    "Rising global temperatures affect agriculture, sea levels, and weather patterns. "
    "Renewable energy sources such as solar and wind power offer pathways to reduce "
    "carbon emissions. International cooperation through treaties and agreements is "
    "essential for coordinated global action.",

    "The human brain contains approximately 86 billion neurons connected by trillions "
    "of synapses. Neural signals travel as electrochemical impulses along axons. "
    "Memory formation involves synaptic strengthening through long-term potentiation. "
    "Sleep plays a crucial role in consolidating memories and clearing metabolic waste.",

    "Economics studies how individuals, firms, and governments allocate scarce resources. "
    "Supply and demand curves determine market prices. Monetary policy by central banks "
    "influences inflation and employment. Fiscal policy through taxation and spending "
    "shapes aggregate demand in an economy.",

    "Photography evolved from daguerreotypes to film to digital sensors. Modern cameras "
    "use CMOS or CCD sensors with millions of pixels. Computational photography applies "
    "machine learning to enhance images automatically. Smartphone cameras now rival "
    "dedicated cameras for most everyday photography needs.",
]

_CODE: List[str] = [
    '''def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted arr, or -1 if absent."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',

    '''class LRUCache:
    """Least-recently-used cache with O(1) get and put."""
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[int, int] = {}
        self.order: list[int] = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)''',

    '''def merge_sort(arr: list[int]) -> list[int]:
    """Recursive merge sort returning a new sorted list."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]''',

    '''import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(
            torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float("-inf")
        )
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)''',

    '''def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Kahn algorithm for topological ordering of a DAG."""
    from collections import deque
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue = deque(n for n, d in in_degree.items() if d == 0)
    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result''',
]

_INSTRUCTION: List[str] = [
    "### Instruction\nExplain the difference between supervised and unsupervised learning.\n"
    "### Response\nSupervised learning trains on labeled data where each input has a known output. "
    "The model learns to map inputs to outputs by minimizing prediction error. Examples include "
    "classification and regression. Unsupervised learning finds patterns in unlabeled data without "
    "predefined targets. Examples include clustering, dimensionality reduction, and generative modeling.",

    "### Instruction\nWhat is gradient descent and why is it used in neural network training?\n"
    "### Response\nGradient descent is an optimization algorithm that iteratively updates model "
    "parameters in the direction that reduces a loss function. At each step, we compute the gradient "
    "of the loss with respect to all parameters and subtract a fraction (learning rate) of that gradient. "
    "This moves parameters toward a local minimum of the loss surface. Neural networks use it because "
    "direct optimization of complex non-convex loss functions is intractable.",

    "### Instruction\nDescribe the transformer architecture briefly.\n"
    "### Response\nThe transformer consists of stacked encoder and decoder blocks, each containing "
    "multi-head self-attention and feed-forward layers with residual connections and layer normalization. "
    "Self-attention computes query, key, value projections and uses scaled dot-product attention to model "
    "dependencies across all positions simultaneously. This replaces recurrence, enabling parallelism "
    "and capturing long-range dependencies more effectively.",

    "### Instruction\nHow does tokenization work in large language models?\n"
    "### Response\nTokenization converts raw text into integer token IDs the model can process. "
    "Modern LLMs use subword tokenization such as Byte-Pair Encoding (BPE) or SentencePiece. "
    "BPE starts with individual characters and iteratively merges the most frequent adjacent pairs "
    "until reaching a target vocabulary size. This balances vocabulary size against sequence length "
    "and handles rare or unseen words gracefully via byte fallback.",

    "### Instruction\nWhat is the purpose of the attention mask in a transformer?\n"
    "### Response\nThe attention mask prevents certain token positions from attending to others. "
    "A causal (autoregressive) mask blocks each position from attending to future positions, "
    "enforcing left-to-right generation. A padding mask blocks attention to pad tokens that were "
    "added to make sequences equal length in a batch. Without masking, the model would leak future "
    "information during training or waste computation attending to meaningless padding.",
]

_MATH: List[str] = [
    "Problem: Find all integer solutions to x² + y² = 25.\n"
    "Solution: We need pairs (x,y) where both are integers and their squares sum to 25. "
    "The perfect squares ≤ 25 are 0, 1, 4, 9, 16, 25. Testing combinations: "
    "0+25=25 gives (0,±5) and (±5,0). 9+16=25 gives (±3,±4) and (±4,±3). "
    "Complete solution set: (±5,0), (0,±5), (±3,±4), (±4,±3) — 12 solutions total.",

    "Problem: Prove that √2 is irrational.\n"
    "Proof by contradiction: Assume √2 = p/q in lowest terms with p,q integers, q≠0. "
    "Then 2 = p²/q², so p² = 2q². This means p² is even, so p is even: p = 2k. "
    "Substituting: 4k² = 2q², so q² = 2k², meaning q is also even. "
    "But then p and q share factor 2, contradicting our assumption of lowest terms. "
    "Therefore √2 is irrational. QED.",

    "Problem: Sum the series 1 + 1/2 + 1/4 + 1/8 + ... to infinity.\n"
    "Solution: This is a geometric series with first term a=1 and ratio r=1/2. "
    "Since |r| < 1, the series converges. Sum = a/(1-r) = 1/(1-1/2) = 1/(1/2) = 2. "
    "We can verify: S_n = 1 - (1/2)^n → 1 as n → ∞, and the full sum doubles this: S = 2.",

    "Problem: A bag contains 3 red and 5 blue balls. Two are drawn without replacement. "
    "What is the probability both are red?\n"
    "Solution: P(first red) = 3/8. Given first is red, P(second red) = 2/7. "
    "P(both red) = (3/8) × (2/7) = 6/56 = 3/28 ≈ 0.107.",

    "Problem: Find the derivative of f(x) = x³ sin(x).\n"
    "Solution: Using the product rule: f'(x) = (x³)' sin(x) + x³ (sin(x))'. "
    "= 3x² sin(x) + x³ cos(x). We can factor: f'(x) = x²(3 sin(x) + x cos(x)).",
]

_EVAL: List[str] = [
    "Evaluation question: What is the capital of France? Answer: Paris.",
    "Evaluation question: What is 7 × 8? Answer: 56.",
    "Evaluation question: Name the first element on the periodic table. Answer: Hydrogen.",
]


def generate_corpus() -> List[Document]:
    """Generate all synthetic documents for all lanes."""
    docs: List[Document] = []

    lane_texts = {
        "english_web": _ENGLISH_WEB,
        "code": _CODE,
        "instruction": _INSTRUCTION,
        "math": _MATH,
    }

    for lane, texts in lane_texts.items():
        for i, text in enumerate(texts):
            docs.append(Document(
                doc_id=f"{lane}_{i:03d}",
                lane=lane,
                text=text,
                is_eval=False,
            ))

    for i, text in enumerate(_EVAL):
        docs.append(Document(
            doc_id=f"eval_{i:03d}",
            lane="eval",
            text=text,
            is_eval=True,
        ))

    return docs
