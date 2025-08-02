import os
from functools import partial

import dspy
from dotenv import load_dotenv
from dspy.datasets import HotPotQA
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.table import Table

load_dotenv()

# Configuration
TRAIN_SIZE = 50
DEV_SIZE = 50
TEST_SIZE = 100
SIMILARITY_THRESHOLD = 0.7

console = Console()
lm = dspy.LM("openrouter/google/gemini-2.0-flash-001", api_key=os.environ["APIKEY"])
dspy.configure(lm=lm)


def cross_encoder_metric(cross_encoder_model, example, pred, trace=None):
    """Metric function using EttinX cross-encoder for semantic similarity evaluation"""
    gold_answer = example.answer.strip()
    pred_answer = pred.answer.strip() if hasattr(pred, "answer") else str(pred).strip()

    sentence_pairs = [(gold_answer, pred_answer)]
    scores = cross_encoder_model.predict(sentence_pairs)
    similarity_score = float(scores[0])

    return (
        similarity_score >= SIMILARITY_THRESHOLD
        if trace is not None
        else similarity_score
    )


def load_dataset(train_size=TRAIN_SIZE, dev_size=DEV_SIZE, test_size=TEST_SIZE):
    """Load HotPotQA dataset with specified splits"""
    console.log(
        f"Loading HotPotQA dataset (train:{train_size}, dev:{dev_size}, test:{test_size})"
    )

    dataset = HotPotQA(
        train_seed=1,
        train_size=train_size,
        eval_seed=2023,
        dev_size=dev_size,
        test_size=test_size,
    )

    # Set input keys as required by DSPy
    splits = {
        "train": [x.with_inputs("question") for x in dataset.train],
        "dev": [x.with_inputs("question") for x in dataset.dev],
        "test": [x.with_inputs("question") for x in dataset.test],
    }

    console.log(
        f"Loaded {len(splits['train'])} train, {len(splits['dev'])} dev, {len(splits['test'])} test examples"
    )
    return splits["train"], splits["dev"], splits["test"]


class QASignature(dspy.Signature):
    """Answer questions clearly and concisely"""

    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="The answer to the question")


def evaluate_program(program, dataset, metric, name="Program"):
    """Evaluate program on dataset using given metric"""
    console.log(f"Evaluating {name}...")

    evaluator = Evaluate(
        devset=dataset, num_threads=1, display_progress=True, display_table=5
    )
    score = evaluator(program, metric=metric)

    console.log(f"{name} Average Score: {score:.4f}")
    return score


def show_example_predictions(
    initial_program, optimized_program, testset, metric, n_examples=3
):
    """Show example predictions from both programs"""
    table = Table(title="Example Predictions")
    table.add_column("Question", style="cyan")
    table.add_column("Gold Answer", style="green")
    table.add_column("Initial Pred", style="yellow")
    table.add_column("Optimized Pred", style="blue")
    table.add_column("Initial Sim", justify="right")
    table.add_column("Optimized Sim", justify="right")

    for example in testset[:n_examples]:
        initial_pred = initial_program(**example.inputs())
        optimized_pred = optimized_program(**example.inputs())

        initial_sim = metric(example, initial_pred)
        optimized_sim = metric(example, optimized_pred)

        table.add_row(
            example.question[:50] + "..."
            if len(example.question) > 50
            else example.question,
            example.answer,
            initial_pred.answer,
            optimized_pred.answer,
            f"{initial_sim:.3f}",
            f"{optimized_sim:.3f}",
        )

    console.print(table)


def main():
    console.rule("[bold red]DSPy HotPotQA with MIPROv2 and EttinX Evaluation")

    # Load components
    console.log("Loading EttinX cross-encoder...")
    cross_encoder = CrossEncoder("dleemiller/EttinX-sts-xs")

    metric = partial(cross_encoder_metric, cross_encoder)
    trainset, devset, testset = load_dataset()

    # Create and evaluate initial program
    console.log("Creating Chain of Thought QA program...")
    # initial_program = dspy.ChainOfThought(QASignature)
    initial_program = dspy.Predict(QASignature)

    console.rule("[bold blue]BEFORE OPTIMIZATION")
    initial_score = evaluate_program(
        initial_program, testset, metric, "Initial Program"
    )

    # Optimize with MIPROv2
    console.rule("[bold green]RUNNING MIPROV2 OPTIMIZATION")
    console.log("Starting MIPROv2 optimization...")

    teleprompter = MIPROv2(metric=metric, auto="light", num_threads=1, verbose=True)

    optimized_program = teleprompter.compile(
        student=initial_program,
        trainset=trainset,
        valset=devset,
        requires_permission_to_run=False,
    )

    # Evaluate optimized program
    console.rule("[bold blue]AFTER OPTIMIZATION")
    optimized_score = evaluate_program(
        optimized_program, testset, metric, "Optimized Program"
    )

    # Results summary
    console.rule("[bold yellow]RESULTS SUMMARY")
    improvement = optimized_score - initial_score
    relative_improvement = (
        (improvement / initial_score * 100) if initial_score > 0 else 0
    )

    results_table = Table(title="Performance Comparison")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Score", justify="right")

    results_table.add_row("Initial Program", f"{initial_score:.4f}")
    results_table.add_row("Optimized Program", f"{optimized_score:.4f}")
    results_table.add_row("Improvement", f"{improvement:+.4f}")
    results_table.add_row("Relative Improvement", f"{relative_improvement:+.2f}%")

    console.print(results_table)

    # Save optimized program
    optimized_program.save("optimized_hotpot_qa.json")
    console.log("Optimized program saved to 'optimized_hotpot_qa.json'")

    # Show example predictions
    console.rule("[bold magenta]EXAMPLE PREDICTIONS")
    show_example_predictions(initial_program, optimized_program, testset, metric)


if __name__ == "__main__":
    main()
