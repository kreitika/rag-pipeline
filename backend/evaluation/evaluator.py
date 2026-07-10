import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from backend.generation.pipeline import full_pipeline

load_dotenv()

QUESTIONS_PATH = Path("evals/golden_qa/questions.json")
RESULTS_PATH   = Path("evals/results.json")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)



def score_correctness(
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> dict:
    prompt = f"""You are evaluating a question-answering system.

Question: {question}

Expected answer: {expected_answer}

System's answer: {actual_answer}

Does the system's answer correctly address the question?
Compare it to the expected answer.

Rate on a scale:
2 = Correct: answer contains the key information from expected answer
1 = Partial: answer contains some but not all key information
0 = Incorrect: answer is wrong, irrelevant, or hallucinates

Respond with only a single digit: 0, 1, or 2.
Then on a new line, one sentence explaining why."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result = response.choices[0].message.content.strip()
    lines = result.split('\n')

    try:
        score = int(lines[0].strip())
        score = max(0, min(2, score))
    except ValueError:
        score = 0

    explanation = lines[1].strip() if len(lines) > 1 else ""

    return {
        "score":       score,
        "max_score":   2,
        "explanation": explanation,
    }


def score_idk_correctness(answer: str, answer_exists: bool) -> dict:
    answer_lower = answer.lower()
    is_idk = (
        "don't have enough information" in answer_lower or
        "cannot find" in answer_lower or
        "not enough information" in answer_lower or
        "no information" in answer_lower
    )

    if not answer_exists:
        if is_idk:
            return {"score": 2, "max_score": 2,
                    "explanation": "Correctly said IDK for unanswerable question"}
        else:
            return {"score": 0, "max_score": 2,
                    "explanation": "Hallucinated answer for unanswerable question"}
    else:
        if is_idk:
            return {"score": 0, "max_score": 2,
                    "explanation": "Said IDK but answer exists in documents"}
        else:
            return {"score": 2, "max_score": 2,
                    "explanation": "Attempted to answer an answerable question"}

def run_eval(max_questions: int = None) -> dict:
    questions = load_questions()

    if max_questions:
        questions = questions[:max_questions]

    results = []
    total_correctness = 0
    total_idk_correct = 0
    answerable_count  = 0
    unanswerable_count = 0

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

        pipeline_result = full_pipeline(q["question"])
        actual_answer   = pipeline_result["answer"]
        confident       = pipeline_result["confident"]

        idk_result = score_idk_correctness(actual_answer, q["answer_exists"])

        if q["answer_exists"]:
            answerable_count += 1
            correctness = score_correctness(
                q["question"],
                q["expected_answer"],
                actual_answer,
            )
            total_correctness += correctness["score"] / correctness["max_score"]
        else:
            unanswerable_count += 1
            correctness = {"score": None, "explanation": "N/A — unanswerable question"}
            if idk_result["score"] == 2:
                total_idk_correct += 1

        result = {
            "id":               q["id"],
            "question":         q["question"],
            "expected_answer":  q["expected_answer"],
            "actual_answer":    actual_answer,
            "answer_exists":    q["answer_exists"],
            "confident":        confident,
            "correctness":      correctness,
            "idk_result":       idk_result,
            "category":         q["category"],
            "difficulty":       q["difficulty"],
        }
        results.append(result)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(1)

    answer_correctness = (
        total_correctness / answerable_count
        if answerable_count > 0 else 0
    )
    idk_accuracy = (
        total_idk_correct / unanswerable_count
        if unanswerable_count > 0 else 0
    )

    summary = {
        "total_questions":    len(questions),
        "answerable":         answerable_count,
        "unanswerable":       unanswerable_count,
        "answer_correctness": round(answer_correctness, 3),
        "idk_accuracy":       round(idk_accuracy, 3),
        "results":            results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("Starting full evaluation — 50 questions")
    print("Estimated time: 15-20 minutes")
    print("Results saved after every question")
    print()

    summary = run_eval(max_questions=None)

    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions:    {summary['total_questions']}")
    print(f"Answerable:         {summary['answerable']}")
    print(f"Unanswerable:       {summary['unanswerable']}")
    print(f"Answer correctness: {summary['answer_correctness']:.1%}")
    print(f"IDK accuracy:       {summary['idk_accuracy']:.1%}")
    print(f"\nResults saved to: {RESULTS_PATH}")