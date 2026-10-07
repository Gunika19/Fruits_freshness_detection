from __future__ import annotations

import argparse

from models.cnn.model import create_dataloaders, evaluate_model, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained fruit freshness CNN.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint.",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Dataset root path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Evaluating on device: {device}")

    model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)

    dataloaders, _, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_size=checkpoint.get("input_size", 224),
        num_workers=args.num_workers,
    )

    confusion, report = evaluate_model(
        model=model,
        dataloader=dataloaders[args.split],
        class_names=class_names,
        device=device,
    )

    print(f"\nCheckpoint model: {checkpoint['model_name']}")
    print(f"Best validation accuracy saved: {checkpoint.get('best_val_acc', 0.0):.4f}")
    print(f"Evaluation split: {args.split}")
    print("\nConfusion Matrix:")
    for row in confusion:
        print(row)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
