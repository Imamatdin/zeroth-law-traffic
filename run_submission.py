"""CLI placeholder. Official serialization must come from the starter kit."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Zeroth Law Traffic submission runner")
    parser.add_argument("--videos", help="Video directory (official contract pending)")
    parser.add_argument("--out", help="Output path (official contract pending)")
    args = parser.parse_args()
    if not args.videos or not args.out:
        parser.error("--videos and --out are required")
    parser.exit(2, "Official starter kit required before generating predictions.\n")


if __name__ == "__main__":
    raise SystemExit(main())
