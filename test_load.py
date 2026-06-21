from connear_pytorch import load_connear


def main():
    model = load_connear("connear/Gmodel.pt", map_location="cuda")
    print(model)


if __name__ == "__main__":
    main()
