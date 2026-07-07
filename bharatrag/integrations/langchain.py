from typing import Any, Optional
from bharatrag import evaluate

try:
    from langchain_core.evaluation import StringEvaluator
except ImportError:
    try:
        from langchain.evaluation import StringEvaluator
    except ImportError:
        try:
            from langchain_classic.evaluation import StringEvaluator
        except ImportError:
            try:
                from langsmith.evaluation import StringEvaluator
            except ImportError:
                raise ImportError(
                    "Could not import StringEvaluator. Please install langchain-core using "
                    "`pip install langchain` to use the Langchain integration."
                )


class BharatRAGLangChainEvaluator(StringEvaluator):
    """BharatRAG Evaluator wrapper for LangCHain.
    Allows running context relevance, groundedness and answer relevance
    evaluations directly inside Langchain workflows.
    """

    def __init__(self, metric: str = "overall", language: str = "hindi", **kwargs: Any):
        """
        Initialize the evaluator.

        Args:
            metric: The specific metric to evaluate ("overall", "context_relevance",
            "groundedness","answer_relevance").

            language: Language of the text ("hindi","marathi","tamil","english").

            **kwargs: Additional arguments to pass to the evaluate function (e.g. groundedness_threshold).

        """

        super().__init__()
        self.metric = metric
        self.language = language
        self.evaluation_kwargs = kwargs

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return True

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the strings.

        Args:
            prediction: The generated answer to evaluate.
            reference: The context retrieved (string or list of strings).
            input: The user query/question.
        """

        if not reference:
            raise ValueError("Reference context is required for evaluation.")

        # Langchain reference might be a single string. Convert to list format for BharatRAG.
        contexts = [reference] if isinstance(reference, str) else list(reference)

        results = evaluate(
            questions=[input or ""],
            contexts=[contexts],
            answers=[prediction],
            language=self.language,
            **self.evaluation_kwargs,
        )

        return {"score": results.get(self.metric, results.get("overall"))}
