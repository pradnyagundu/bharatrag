# Quick test — run this to see if embeddings work on Hindi text
from bharatrag.embeddings.indic_embeddings import IndicEmbedder

# Load Hindi model
embedder = IndicEmbedder(language="hindi")

# Test 1 — Similar sentences (should score HIGH)
score1 = embedder.similarity(
    "भारत की राजधानी क्या है?",
    "भारत की राजधानी नई दिल्ली है।"
)
print(f"Similar sentences score: {score1:.4f}  (should be HIGH > 0.7)")

# Test 2 — Unrelated sentences (should score LOW)
score2 = embedder.similarity(
    "भारत की राजधानी क्या है?",
    "आज मौसम बहुत अच्छा है।"
)
print(f"Unrelated sentences score: {score2:.4f}  (should be LOW < 0.4)")

# Test 3 — One question vs many chunks
chunks = [
    "भारत की राजधानी नई दिल्ली है।",          # relevant
    "मुंबई भारत की आर्थिक राजधानी है।",        # somewhat relevant  
    "आज क्रिकेट मैच बहुत अच्छा था।",           # not relevant
]
scores = embedder.similarity_one_to_many(
    "भारत की राजधानी क्या है?",
    chunks
)
print(f"\nOne-to-many scores:")
for chunk, score in zip(chunks, scores):
    print(f"  {score:.4f} — {chunk}")





# Test Context Relevance Metric
print("\n" + "="*50)
print("TESTING CONTEXT RELEVANCE METRIC")
print("="*50)

from bharatrag.metrics.context_relevance import ContextRelevance

cr = ContextRelevance(language="hindi")


# Test 1 — Relevant context (should score HIGH)
score1 = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=[
        "भारत की राजधानी नई दिल्ली है।",
        "नई दिल्ली भारत का सबसे बड़ा शहर है।"
    ]
)
print(f"\nTest 1 - Relevant context: {score1}  (should be HIGH > 0.5)")

# Test 2 — Irrelevant context (should score LOW)
score2 = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=[
        "आज मौसम बहुत अच्छा है।",
        "क्रिकेट भारत का लोकप्रिय खेल है।"
    ]
)
print(f"Test 2 - Irrelevant context: {score2}  (should be LOW < 0.3)")

# Test 3 — Detailed breakdown
print("\nTest 3 - Detailed breakdown:")
details = cr.score_detailed(
    question="पीएम किसान योजना क्या है?",
    contexts=[
        "पीएम किसान सम्मान निधि योजना भारत सरकार की एक योजना है।",
        "इस योजना के तहत किसानों को हर साल 6000 रुपये मिलते हैं।",
        "आज बारिश हो रही है।"
    ]
)
print(f"Overall score: {details['overall']}")
for chunk in details['chunks']:
    print(f"  {chunk['score']} — {chunk['chunk']}")






    # Test Groundedness Metric
print("\n" + "="*50)
print("TESTING GROUNDEDNESS METRIC")
print("="*50)

from bharatrag.metrics.groundedness import Groundedness

gr = Groundedness(language="hindi")

# Test 1 — Fully grounded answer (should score HIGH)
score1 = gr.score(
    answer="भारत की राजधानी नई दिल्ली है। यह एक बड़ा शहर है।",
    contexts=[
        "भारत की राजधानी नई दिल्ली है।",
        "नई दिल्ली भारत का सबसे बड़ा शहर है।"
    ]
)
print(f"\nTest 1 - Grounded answer: {score1}  (should be HIGH > 0.7)")

# Test 2 — Hallucinated answer (should score LOW)
score2 = gr.score(
    answer="भारत की राजधानी मुंबई है। वहां 5 करोड़ लोग रहते हैं।",
    contexts=[
        "भारत की राजधानी नई दिल्ली है।",
        "नई दिल्ली में संसद भवन है।"
    ]
)
print(f"Test 2 - Hallucinated answer: {score2}  (should be LOW < 0.5)")

# Test 3 — Detailed breakdown
print("\nTest 3 - Detailed breakdown:")
details = gr.score_detailed(
    answer="पीएम किसान योजना किसानों के लिए है। इसमें 6000 रुपये मिलते हैं। यह योजना 2019 में शुरू हुई।",
    contexts=[
        "पीएम किसान सम्मान निधि योजना भारत सरकार की किसानों के लिए योजना है।",
        "इस योजना के तहत किसानों को हर साल 6000 रुपये तीन किश्तों में मिलते हैं।",
    ]
)
print(f"Overall: {details['overall']}")
print(f"Supported: {details['supported']}/{details['total_claims']} claims")
for claim in details['claims']:
    status = "✅ supported" if claim['supported'] else "❌ hallucinated"
    print(f"  {status} ({claim['best_similarity']}) — {claim['claim']}")










# Test Answer Relevance Metric
print("\n" + "="*50)
print("TESTING ANSWER RELEVANCE METRIC")
print("="*50)

from bharatrag.metrics.answer_relevance import AnswerRelevance

ar = AnswerRelevance(language="hindi")

# Test 1 — Relevant answer (should score HIGH)
score1 = ar.score(
    question="भारत की राजधानी क्या है?",
    answer="भारत की राजधानी नई दिल्ली है।"
)
print(f"\nTest 1 - Relevant answer: {score1}  (should be HIGH > 0.5)")

# Test 2 — Irrelevant answer (should score LOW)
score2 = ar.score(
    question="भारत की राजधानी क्या है?",
    answer="आज मौसम बहुत अच्छा है और बारिश हो रही है।"
)
print(f"Test 2 - Irrelevant answer: {score2}  (should be LOW < 0.3)")

# Test 3 — Detailed breakdown
print("\nTest 3 - Detailed breakdown:")
cases = [
    ("पीएम किसान योजना क्या है?", "पीएम किसान सम्मान निधि एक सरकारी योजना है जो किसानों को 6000 रुपये देती है।"),
    ("पीएम किसान योजना क्या है?", "नई दिल्ली भारत की राजधानी है।"),
    ("पीएम किसान योजना क्या है?", "मुझे नहीं पता।"),
]
for q, a in cases:
    detail = ar.score_detailed(q, a)
    print(f"  {detail['overall']} ({detail['interpretation']}) — {detail['answer']}")




# Test the full evaluate() function
print("\n" + "="*50)
print("TESTING FULL BHARATRAG EVALUATE()")
print("="*50)

from bharatrag import evaluate

results = evaluate(
    questions=[
        "भारत की राजधानी क्या है?",
        "पीएम किसान योजना क्या है?",
    ],
    contexts=[
        ["भारत की राजधानी नई दिल्ली है।",
         "नई दिल्ली में संसद भवन है।"],
        ["पीएम किसान सम्मान निधि किसानों के लिए योजना है।",
         "इस योजना में 6000 रुपये सालाना मिलते हैं।"],
    ],
    answers=[
        "भारत की राजधानी नई दिल्ली है।",
        "पीएम किसान योजना में किसानों को 6000 रुपये मिलते हैं।",
    ],
    language="hindi"
)

print("\n📊 BharatRAG Evaluation Results:")
print(f"  Context Relevance:  {results['context_relevance']}")
print(f"  Groundedness:       {results['groundedness']}")
print(f"  Answer Relevance:   {results['answer_relevance']}")
print(f"  ─────────────────────────────")
print(f"  Overall Score:      {results['overall']}")
print(f"  Language:           {results['language']}")
print(f"  Questions scored:   {results['num_questions']}")





# Test Marathi Support
print("\n" + "="*50)
print("TESTING MARATHI SUPPORT")
print("="*50)

from bharatrag import evaluate

results = evaluate(
    questions=[
        "भारताची राजधानी कोणती आहे?",
        "पीएम किसान योजना काय आहे?",
    ],
    contexts=[
        ["भारताची राजधानी नवी दिल्ली आहे.",
         "नवी दिल्ली हे भारतातील सर्वात मोठे शहर आहे."],
        ["पीएम किसान सन्मान निधी ही शेतकऱ्यांसाठी योजना आहे.",
         "या योजनेत शेतकऱ्यांना दरवर्षी 6000 रुपये मिळतात."],
    ],
    answers=[
        "भारताची राजधानी नवी दिल्ली आहे.",
        "पीएम किसान योजनेत शेतकऱ्यांना 6000 रुपये मिळतात.",
    ],
    language="marathi"
)

print("\n📊 BharatRAG Marathi Evaluation Results:")
print(f"  Context Relevance:  {results['context_relevance']}")
print(f"  Groundedness:       {results['groundedness']}")
print(f"  Answer Relevance:   {results['answer_relevance']}")
print(f"  ─────────────────────────────")
print(f"  Overall Score:      {results['overall']}")
print(f"  Language:           {results['language']}")