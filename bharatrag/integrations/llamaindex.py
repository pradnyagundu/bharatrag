from typing import Any, Optional, Sequence, Dict
from bharatrag import evaluate

try:
    from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
except ImportError:
    raise ImportError(
        "Could not import llama_index. Please install it using "
        "`pip install llama-index` to use the LLamaIndex integration. "
    )


class BharatRAGLlamaIndexEvaluator(BaseEvaluator):
    """BharatRAG Evaluator wrapper for LlamaIndex."""

    def __init__(self, metric: str = "overall", language: str = "hindi", **kwargs: Any):
        """Initialize the evaluator.

        Args:
            metric: The specific metric to evaluate ("overall", "context_relevance", "groundedness", "answer_relevance")
            language: Language of the text (see bharatrag.SUPPORTED_LANGUAGES
                      for the full list).
            **kwargs: Addtional arguments to pass to the evaluate function (e.g. groundedness_threshold).
        """

        self.metric = metric
        self.language = language
        self.evaluation_kwargs = kwargs
    
    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts"""
        return {}
    
    def _update_prompts(self, prompts: Dict[str, Any]) -> None:
        """Update prompts."""
        pass

    def evaluate(
        self,
        query: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        response: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the response against the query and contexts."""
        if not query or not contexts or not response:
            return EvaluationResult(
                query=query,
                contexts=contexts,
                response=response,
                passing=False,
                feedback="Missing query, contexts or response.",
            )

        results = evaluate(
            questions=[query],
            contexts=[list(contexts)],
            answers=[response],
            language=self.language,
            **self.evaluation_kwargs,
        )

        score = results.get(self.metric, results.get("overall"))

        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            score=score,
            passing=score >= 0.5,  # Default threshold for a passing evaluation.
            feedback=f"BharatRAG {self.metric} score: {score}",
        )

    async def aevaluate(
        self,
        query: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        response: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Asynchronous evaluation (calls the synchronous evaluate under the hood)"""
        return self.evaluate(
            query=query, contexts=contexts, response=response, **kwargs
        )
