"""Convert the original CoNNear Keras weights to a PyTorch state dict."""

from __future__ import annotations

import argparse

from connear_pytorch import save_converted_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        default="connear/Gmodel.h5",
        help="Path to the original Keras .h5 weights",
    )
    parser.add_argument(
        "--out",
        default="connear/Gmodel.pt",
        help="Path for the converted PyTorch weights",
    )
    args = parser.parse_args()

    output_path = save_converted_weights(args.h5, args.out)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
