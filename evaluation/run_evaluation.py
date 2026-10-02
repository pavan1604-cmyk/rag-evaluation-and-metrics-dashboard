import sys
import json
from pathlib import Path

import pandas as pd

from ragas import EvaluationDataset, evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from src.rag_pipeline import RAGPipeline
from src.ragas_llm import GroqRagasLLM
from src.ragas_embeddings import SentenceTransformerRagasEmbeddings


PDF_PATH = PROJECT_ROOT / "data" / "company_report.pdf"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "evaluation_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "evaluation_results.csv"


def main():

    print("=" * 70)
    print("RAG EVALUATION PIPELINE")
    print("=" * 70)

    print("\nLoading evaluation dataset...")

    dataset = pd.read_csv(DATASET_PATH)

    print(f"Evaluation questions: {len(dataset)}")

    # ---------------------------------------------------------
    # Step 1: Initialize RAG pipeline
    # ---------------------------------------------------------

    print("\nInitializing RAG pipeline...")

    pipeline = RAGPipeline(
        pdf_path=str(PDF_PATH),
        chunk_size=800,
        overlap=150,
        candidate_k=8,
        top_k=5
    )

    # ---------------------------------------------------------
    # Step 2: Generate RAG answers
    # ---------------------------------------------------------

    results = []

    for index, row in dataset.iterrows():

        question = row["question"]
        ground_truth = row["ground_truth"]

        print("\n" + "=" * 70)
        print(f"Question {index + 1}/{len(dataset)}")
        print("=" * 70)

        print(question)

        result = pipeline.query(question)

        answer = result["answer"]

        print("\nGenerated Answer:")
        print(answer)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "retrieved_contexts": result["contexts"]
        })

    # ---------------------------------------------------------
    # Step 3: Create evaluation DataFrame
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RAG GENERATION COMPLETE")
    print("=" * 70)

    print(f"Generated answers: {len(results_df)}")

    # ---------------------------------------------------------
    # Step 4: Prepare data for RAGAS
    # ---------------------------------------------------------

    ragas_df = results_df.copy()

    ragas_df = ragas_df.rename(
        columns={
            "question": "user_input",
            "answer": "response",
            "ground_truth": "reference"
        }
    )

    # ---------------------------------------------------------
    # Step 5: Create RAGAS dataset
    # ---------------------------------------------------------

    print("\nCreating RAGAS evaluation dataset...")

    evaluation_dataset = EvaluationDataset.from_pandas(
        ragas_df[
            [
                "user_input",
                "response",
                "reference",
                "retrieved_contexts"
            ]
        ]
    )

    # ---------------------------------------------------------
    # Step 6: Initialize RAGAS LLM and embeddings
    # ---------------------------------------------------------

    print("\nLoading RAGAS evaluation models...")

    ragas_llm = GroqRagasLLM()

    ragas_embeddings = SentenceTransformerRagasEmbeddings()

    # ---------------------------------------------------------
    # Step 7: Define evaluation metrics
    # ---------------------------------------------------------

    metrics = [
        Faithfulness(
            llm=ragas_llm
        ),

        AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings
        ),

        ContextPrecision(
            llm=ragas_llm
        ),

        ContextRecall(
            llm=ragas_llm
        )
    ]

    # ---------------------------------------------------------
    # Step 8: Run RAGAS evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 70)

    ragas_result = evaluate(
        evaluation_dataset,
        metrics=metrics,
        embeddings=ragas_embeddings,
        raise_exceptions=True
    )

    print("\nRAGAS evaluation completed.")

    # ---------------------------------------------------------
    # Step 9: Convert RAGAS results to DataFrame
    # ---------------------------------------------------------

    metric_results = ragas_result.to_pandas()

    # ---------------------------------------------------------
    # Step 10: Add metric scores to results
    # ---------------------------------------------------------

    results_df["faithfulness"] = metric_results[
        "faithfulness"
    ].values

    results_df["answer_relevancy"] = metric_results[
        "answer_relevancy"
    ].values

    results_df["context_precision"] = metric_results[
        "context_precision"
    ].values

    results_df["context_recall"] = metric_results[
        "context_recall"
    ].values

    # ---------------------------------------------------------
    # Step 11: Store retrieved contexts as JSON
    # ---------------------------------------------------------

    results_df["retrieved_contexts"] = results_df[
        "retrieved_contexts"
    ].apply(json.dumps)

    # ---------------------------------------------------------
    # Step 12: Save evaluation results
    # ---------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Step 13: Display summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)

    print(f"\nRows written: {len(results_df)}")

    print("\nMetric Summary:")

    print(
        f"Faithfulness:      "
        f"{results_df['faithfulness'].mean():.4f}"
    )

    print(
        f"Answer Relevancy:  "
        f"{results_df['answer_relevancy'].mean():.4f}"
    )

    print(
        f"Context Precision: "
        f"{results_df['context_precision'].mean():.4f}"
    )

    print(
        f"Context Recall:    "
        f"{results_df['context_recall'].mean():.4f}"
    )


if __name__ == "__main__":
    main()