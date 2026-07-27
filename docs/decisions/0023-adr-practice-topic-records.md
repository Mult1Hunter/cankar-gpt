# ADR 0023 - ADR practice: topic records, proposed not emitted

**Status:** accepted, 2026-07-26

## Context

The set reached 22 records, 5 of them written in a single session - 22% in one
day. An audit against the bar "would a competent developer joining in a year ask
*why on earth did they do it this way?*" found 11 genuine decisions. The inflation
was not spread out: it sat entirely in the process/meta band, where **every record
was duplicated by the mechanism it created** (ADR 0008 had five copies - the rules
file, an AST test, mypy and import-linter config, `cankar/core/` itself, and the
design-brief trigger table).

Two sentences caused it. CLAUDE.md's ratchet read "convert every caught mistake
into a gate, **then record the ADR**" - a pump with no throttle; and "structure
changes... cite an ADR" read in practice as *write one*.

Retrieval was measured, not assumed: `graphify` indexes ADR titles and `##`
headings only. Body prose is invisible - a query for "squash merge policy" misses
the ADR that decided it. And CLAUDE.md never told a session to consult
`docs/decisions/` at all, so the records documented drift instead of preventing it.

## Decision

1. **Records are topic-based and amended in place.** A changed decision appends a
   dated `## Amendment`; a new number only for a genuinely new topic.
2. **ADRs are proposed, never emitted.** The `adr` skill states a three-part bar -
   hard to reverse *and* a nameable rejected alternative *and* a cold session would
   otherwise redo it - then pitches one line and stops. It carries a routing table
   for everything that fails the bar.
3. **The gate's docstring is the record.** CLAUDE.md's ratchet is rewritten; an ADR
   only when the choice of gate had a real alternative.
4. **`docs/decisions/README.md` is the entry point**, one line per record written as
   the question it answers, gated for completeness and supersession symmetry by
   `tests/structure/test_adr_index.py`.
5. **Two outcomes, distinguished by where the content now lives.** A record still
   authoritative but partly overtaken **keeps its body** and appends to its Status
   only. A record whose content has **moved** becomes a tombstone: body replaced by
   a stub, number retained, Status carrying `-> <destination>`. `->` is the single
   sentinel, in the record and in its index row, and it is gated both ways.

## Rationale

- **Numbers are permanent.** Hundreds of `ADR NNNN` citations exist across tracked
  files (30 point at 0016 alone), and the contiguity check forbids gaps. Deletion
  would break every one; a five-line stub costs nothing and documents the demotion.
- **Append to Status, never rewrite the body.** The two prior conventions
  contradicted each other. A reader landing cold on a half-dead record via grep or
  a code citation must see it, and the index's Relationship column is provably
  invisible to the retriever (zero graph nodes for the table).
- **Do not gate length.** A line-count test rewards reflowing prose into fewer,
  longer lines, and the overflow is the case-study material - ADR 0006's ~150
  amputated poems is why the calibration rule gets obeyed. Shape is advisory, and
  the human at review is the check.
- **A rule without a check is a wish UNLESS a human is the check.** "Was this hard
  to reverse?" is unjudgeable by a machine and every proxy is self-satisfied by the
  session that wants the ADR. Every PR is human-merged and a new file under
  `docs/decisions/` is maximally visible in a diff. That is the check - stated
  explicitly so the next session does not invent a gate for it.

## Consequences

- 22 records become 11 live topic records plus 11 tombstones. Case studies were
  carried to their destinations, but a design-review pass proved a first pass had
  silently dropped several - the closure verification behind the headline BPB
  table, the eval-side defense-in-depth guard, the silenced-SessionStart
  rationale, the Pages `build_type` fact. **Content loss is the real cost of
  consolidating, and it does not announce itself.** Anything demoted in future gets
  its destination checked, not assumed.
- Path-scoped `.claude/rules/` files carry the pointers, so the relevant record
  surfaces automatically where it applies at zero resident context cost.
- Deferred: nothing enforces that an ADR is cited from a non-`docs/` file. It holds
  for all of them today and a comment would game it - recorded, not built.
