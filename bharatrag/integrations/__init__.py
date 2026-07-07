"""
BharatRAG framework integrations.

Evaluators are imported lazily so that installing the core library
does not require langchain or llama-index. Import the specific
evaluator you need:

    from bharatrag.integrations import BharatRAGLangChainEvaluator
    from bharatrag.integrations import BharatRAGLlamaIndexEvaluator
"""

__all__ = ["BharatRAGLangChainEvaluator", "BharatRAGLlamaIndexEvaluator"]


def __getattr__(name):
    if name == "BharatRAGLangChainEvaluator":
        from bharatrag.integrations.langchain import BharatRAGLangChainEvaluator
        return BharatRAGLangChainEvaluator
    if name == "BharatRAGLlamaIndexEvaluator":
        from bharatrag.integrations.llamaindex import BharatRAGLlamaIndexEvaluator
        return BharatRAGLlamaIndexEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")