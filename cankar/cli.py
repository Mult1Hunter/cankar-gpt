"""The single console entry point: `cankar <stage> <command>` (ADR 0007).

Each stage package owns exactly one cli.py that registers its subcommands
here. There is no scripts/ directory; a "quick script" is a stage module
with a subcommand.
"""

from __future__ import annotations

import argparse
import logging
import sys

from cankar.core.errors import CankarError
from cankar.corpus import cli as corpus_cli
from cankar.evals import cli as evals_cli
from cankar.tokenizer import cli as tokenizer_cli
from cankar.train import cli as train_cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cankar", description="CankarGTP pipeline CLI")
    stages = ap.add_subparsers(dest="stage", required=True)
    corpus_cli.register(stages.add_parser("corpus", help="Phase 1: corpus acquisition"))
    tokenizer_cli.register(stages.add_parser("tokenizer", help="Phase 2: Slovene BPE"))
    evals_cli.register(stages.add_parser("evals", help="Phase 2.25: eval harness"))
    train_cli.register(stages.add_parser("train", help="Phase 2.5/3: train the GPT"))
    # future stages register here: pairs (Ph5)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        return int(args.func(args))
    except CankarError as exc:
        logging.getLogger("cankar").error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
