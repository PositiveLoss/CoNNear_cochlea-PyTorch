import torch

from connear_pytorch import load_connear


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_connear("connear/Gmodel.pt", map_location=device)
    print(model)


if __name__ == "__main__":
    main()
