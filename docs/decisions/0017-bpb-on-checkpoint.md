# ADR 0017 - held-out BPB on a trained checkpoint

**Status:** merged into ADR 0016 -> docs/decisions/0016-training-stage.md (2026-07-26)

## Context

The self-describing-checkpoint constraint (evals reads gptconfig, never imports train) is three bullets in the training record.

## Decision

Withdrawn as a standing record. The content lives at `docs/decisions/0016-training-stage.md`; this number
is retained so existing `ADR 0017` citations still resolve.

## Consequences

Nothing changes mechanically - see ADR 0023 for the consolidation that produced
this stub, and `docs/decisions/README.md` for where each topic now lives.
