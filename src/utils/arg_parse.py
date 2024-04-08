import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    args = parser.parse_args()

    return args
