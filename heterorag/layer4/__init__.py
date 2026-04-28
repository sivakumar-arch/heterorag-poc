# heterorag/layer4/__init__.py
"""
Layer 4 — Generation

Public API:
    GenerationLLM    — Anthropic API wrapper for answer generation
    GenerationResult — dataclass returned by GenerationLLM.generate()
    HeteroRAGPipeline — full four-layer pipeline (I₁ → GenerationResult)
"""
from .generation import GenerationLLM, GenerationResult
from .pipeline import HeteroRAGPipeline

__all__ = [
    "GenerationLLM",
    "GenerationResult",
    "HeteroRAGPipeline",
]
