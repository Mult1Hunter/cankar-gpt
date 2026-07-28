---
name: design-brief
description: Architect step before non-trivial code - a short design brief with enumerated data classes, failure modes, and a validation plan. Use BEFORE implementing any new module, script, heuristic, or pipeline stage. Trivial edits are exempt.
---

# Design brief (the architect step)

Post this brief in-chat BEFORE writing the code. It exists because foresight
failures are this project's dominant defect class (ADR 0006: a bibliography
detector shipped against synthetic tests and amputated ~150 poems - one
enumeration question would have caught it).

## The brief (~10 lines, no ceremony)

1. **Goal** - one sentence.
2. **Prior art** - name the STANDARD approach for this class of problem, checked
   and cited, not recalled. Then say which one you are taking and what you are
   rejecting. Skip only when there is no established practice to check.
   This line exists because Phase 6 shipped a full fine-tune with no rehearsal -
   the option with the worst known forgetting behaviour - and then spent a
   learning-rate sweep managing the damage. The implementation details were
   researched; the *approach* was assumed. One search would have caught it
   (2026-07-28).
3. **Inputs/outputs** - types, schemas, where artifacts land.
4. **Data classes** - ENUMERATE every kind of input this code will meet
   (for corpus text: prose, verse, drama, catalogs/indexes, OCR noise,
   mixed-language, ...). This line is the whole point: name the class you
   have not thought about yet.
5. **Failure modes + blast radius** - what breaks if this is wrong, and what
   it contaminates downstream (e.g. misattribution poisons the style
   classifier).
6. **Placement** - which stage subpackage (ADR 0007); what
   existing code it reuses.
7. **Validation plan** - what proves it works BEFORE it ships. Heuristics and
   thresholds: the calibration rule (ADR 0006) applies - labeled REAL examples
   of every enumerated class, which then become regression fixtures.
8. **Out of scope** - what this deliberately does not handle, recorded.

## Architect critique (new-subsystem scale)

For phase kickoffs, new pipeline stages, or anything establishing a pattern
others will follow: after drafting the brief, spawn a Plan/architect agent to
challenge it (fresh eyes, read-only) and address its must-fix points before
implementing. Solo-file changes inside an established pattern do not need the
agent - the brief alone suffices.

## Decision triggers (reach for the pattern PROACTIVELY)

| When you see | Reach for |
|---|---|
| a closed set of string values | StrEnum (serialization stays plain strings) |
| a dict with fixed magic keys | dataclass (frozen if value-like) or pydantic model |
| a SECOND similar implementation | protocol/shared abstraction - or a written deferral |
| branching on a type/kind string | polymorphism via protocol dispatch |
| a boolean parameter that switches behavior | two functions, or an enum mode |
| raw dict access into config/TOML/JSON | a validated pydantic model at the boundary |
| a magic number in logic | named constant; calibration provenance if it is a threshold (ADR 0006) |
| I/O + policy + formatting in one function | split by altitude |
| error signaling via exit/print in a module | domain exception (core/errors.py) + logging |
| a relative path literal | core/paths.py helper |

Counterweight: patterns must earn their existence - no protocols with one
implementer, no config for constants. Rule of two, not rule of maybe-someday.

Written deferrals live as ROADMAP items (status home), not in this skill.

## Exemptions

Config tweaks, doc edits, mechanical renames, changes fully covered by an
existing brief. When in doubt, write the brief - it costs a minute.
