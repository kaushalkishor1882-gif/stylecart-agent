# ============================================================
# PART 6: RAGAS BASELINE EVALUATION
# ============================================================


from part4_5 import ask, app
from part2_3 import embedder, collection

# ── 5 evaluation question-answer pairs with ground truth ─────
eval_pairs = [
    {
        "question": "What is the return window for StyleCart?",
        "ground_truth": "StyleCart accepts returns within 7 days of delivery. Items must be unused, unwashed, and in original condition with all tags attached."
    },
    {
        "question": "How long does standard delivery take?",
        "ground_truth": "Standard delivery takes 3 to 5 business days from the date of dispatch. Express delivery takes 2 business days for an extra Rs. 99."
    },
    {
        "question": "Is Cash on Delivery available and what is the handling fee?",
        "ground_truth": "Cash on Delivery is available for orders up to Rs. 5,000 at serviceable pin codes. A handling fee of Rs. 50 is charged on all COD orders."
    },
    {
        "question": "How can I earn StyleCoins?",
        "ground_truth": "Earn 1 StyleCoin for every Rs. 10 spent on prepaid orders. COD orders earn StyleCoins only after successful delivery. 100 StyleCoins equal Rs. 10 discount."
    },
    {
        "question": "How do I cancel my order?",
        "ground_truth": "Orders can be cancelled within 2 hours of placing, provided they are not yet dispatched. Go to My Orders and click Cancel Order."
    }
]


def run_ragas_evaluation():
    """Run RAGAS evaluation on the 5 test pairs."""

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("Running agent for each evaluation question...\n")

    for i, pair in enumerate(eval_pairs):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        print(f"Eval Q{i+1}: {question}")

        # Run the agent
        result = ask(question, thread_id=f"ragas_eval_{i}")
        answer = result.get("answer", "")
        retrieved = result.get("retrieved", "")

        # Collect context chunks as a list of strings
        contexts = []
        if retrieved:
            chunks = retrieved.split("\n\n")
            contexts = [c for c in chunks if c.strip()]

        print(f"  Answer: {answer[:80]}...")
        print(f"  Contexts count: {len(contexts)}")

        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(ground_truth)

    # ── Try RAGAS evaluation ──────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        print("\nRunning RAGAS evaluation...")
        dataset = Dataset.from_dict(eval_data)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )

        print("\n" + "="*50)
        print("RAGAS BASELINE SCORES")
        print("="*50)
        print(f"Faithfulness     : {scores['faithfulness']:.3f}")
        print(f"Answer Relevancy : {scores['answer_relevancy']:.3f}")
        print(f"Context Precision: {scores['context_precision']:.3f}")

        avg = (scores['faithfulness'] + scores['answer_relevancy'] + scores['context_precision']) / 3
        print(f"\nAverage Score    : {avg:.3f}")

        if scores['faithfulness'] >= 0.70:
            print("✅ Faithfulness: PASS (>= 0.70)")
        else:
            print("❌ Faithfulness: FAIL — check system prompt grounding rules")

        return scores

    except ImportError:
        # RAGAS not installed — use manual LLM faithfulness scoring
        print("\n⚠️  RAGAS not installed. Using manual faithfulness scoring as fallback.")
        run_manual_faithfulness(eval_data)


def run_manual_faithfulness(eval_data):
    """Fallback: score faithfulness manually using LLM."""
    from part2_3 import llm

    print("\n" + "="*50)
    print("MANUAL FAITHFULNESS SCORES (RAGAS fallback)")
    print("="*50)

    scores = []
    for i, (q, a, ctxs, gt) in enumerate(zip(
        eval_data["question"],
        eval_data["answer"],
        eval_data["contexts"],
        eval_data["ground_truth"]
    )):
        context_text = "\n".join(ctxs)

        prompt = f"""Score whether this ANSWER is faithful to the CONTEXT.
Faithful means every fact in the answer comes from the context only.

Score: 0.0 (completely unfaithful) to 1.0 (completely faithful)

CONTEXT: {context_text}
ANSWER: {a}

Reply with a number only between 0.0 and 1.0:"""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5

        scores.append(score)
        status = "PASS" if score >= 0.70 else "FAIL"
        print(f"Q{i+1}: {q[:50]:<50} Score={score:.2f}  {status}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Faithfulness: {avg_score:.3f}")
    if avg_score >= 0.70:
        print("✅ Overall faithfulness: PASS")
    else:
        print("❌ Overall faithfulness: FAIL — improve system prompt grounding")

    return scores


if __name__ == "__main__":
    print("="*60)
    print("PART 6: RAGAS BASELINE EVALUATION")
    print("="*60)
    run_ragas_evaluation()
    print("\n✅ Part 6 complete. Record these scores in your written summary.")
