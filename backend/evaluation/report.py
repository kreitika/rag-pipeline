import json
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("evals/results.json")


def generate_report() -> dict:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    results = data["results"]

    # --- Answer Correctness ---
    answerable = [r for r in results if r["answer_exists"]]
    correctness_scores = []
    for r in answerable:
        if r["correctness"]["score"] is not None:
            score = r["correctness"]["score"] / r["correctness"]["max_score"]
            correctness_scores.append(score)

    answer_correctness = (
        sum(correctness_scores) / len(correctness_scores)
        if correctness_scores else 0
    )

    # --- IDK Accuracy ---
    unanswerable = [r for r in results if not r["answer_exists"]]
    idk_correct = sum(
        1 for r in unanswerable
        if r["idk_result"]["score"] == 2
    )
    idk_accuracy = idk_correct / len(unanswerable) if unanswerable else 0

    # --- Citation Accuracy ---
    citation_scores = []
    for r in answerable:
        if "verification" in r and r["verification"]:
            citation_scores.append(r["verification"]["citation_score"])

    citation_accuracy = (
        sum(citation_scores) / len(citation_scores)
        if citation_scores else 0
    )

    # --- Confidence Score Distribution ---
    confident_count = sum(1 for r in results if r["confident"])
    not_confident_count = len(results) - confident_count

    # --- By Category ---
    category_scores = defaultdict(list)
    for r in answerable:
        if r["correctness"]["score"] is not None:
            score = r["correctness"]["score"] / r["correctness"]["max_score"]
            category_scores[r["category"]].append(score)

    category_report = {}
    for cat, scores in category_scores.items():
        category_report[cat] = round(sum(scores) / len(scores), 3)

    # --- By Difficulty ---
    difficulty_scores = defaultdict(list)
    for r in answerable:
        if r["correctness"]["score"] is not None:
            score = r["correctness"]["score"] / r["correctness"]["max_score"]
            difficulty_scores[r["difficulty"]].append(score)

    difficulty_report = {}
    for diff, scores in difficulty_scores.items():
        difficulty_report[diff] = round(sum(scores) / len(scores), 3)

    return {
        "total_questions":    len(results),
        "answerable":         len(answerable),
        "unanswerable":       len(unanswerable),
        "answer_correctness": round(answer_correctness, 3),
        "idk_accuracy":       round(idk_accuracy, 3),
        "citation_accuracy":  round(citation_accuracy, 3),
        "confident_count":    confident_count,
        "not_confident_count": not_confident_count,
        "by_category":        category_report,
        "by_difficulty":      difficulty_report,
    }



if __name__ == "__main__":
    report = generate_report()

    print("=" * 60)
    print("RAG PIPELINE EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOVERALL METRICS:")
    print(f"  Total questions:    {report['total_questions']}")
    print(f"  Answer correctness: {report['answer_correctness']:.1%}")
    print(f"  IDK accuracy:       {report['idk_accuracy']:.1%}")
    print(f"  Citation accuracy:  {report['citation_accuracy']:.1%}")

    print(f"\nCONFIDENCE DISTRIBUTION:")
    print(f"  Confident answers:  {report['confident_count']}")
    print(f"  IDK responses:      {report['not_confident_count']}")

    print(f"\nBY CATEGORY:")
    for cat, score in sorted(report['by_category'].items()):
        bar = "█" * int(score * 20)
        print(f"  {cat:<12} {score:.1%}  {bar}")

    print(f"\nBY DIFFICULTY:")
    for diff, score in sorted(report['by_difficulty'].items()):
        bar = "█" * int(score * 20)
        print(f"  {diff:<12} {score:.1%}  {bar}")

    print(f"\n{'='*60}")
    print("PORTFOLIO NUMBERS:")
    print(f"{'='*60}")
    print(f"  Answer correctness: {report['answer_correctness']:.1%}")
    print(f"  IDK accuracy:       {report['idk_accuracy']:.1%}")
    print(f"  Citation accuracy:  {report['citation_accuracy']:.1%}")
    print(f"  Evaluated on:       {report['total_questions']} questions")