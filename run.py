import argparse

from production_app.logging_config import configure_logging, get_logger
from production_app.main import predict_workflow, train_workflow

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line interface parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Breast Cancer Classification Production Application"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "train",
        help="Train, evaluate, and save the model.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load the saved model and make a prediction.",
    )

    predict_parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index used for prediction.",
    )

    return parser


def main() -> None:
    """
    Application entry point.
    """
    configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            metrics = train_workflow()

            print("\nTraining completed successfully.")
            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1 Score : {metrics['f1_score']:.4f}")

            print("\nConfusion Matrix:")
            print(metrics["confusion_matrix"])

            print("\nClassification Report:")
            print(metrics["classification_report"])

        elif args.command == "predict":
            prediction = predict_workflow(
                sample_index=args.sample_index,
            )

            print(f"\nPrediction for sample {args.sample_index}: " f"{prediction}")

    except (FileNotFoundError, ValueError) as exc:
        logger.error(
            "Application failed: %s",
            exc,
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
