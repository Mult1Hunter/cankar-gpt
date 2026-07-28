#!/usr/bin/env bash
# Mutation harness: prove a gate fails when the thing it guards is broken.
#
# A test that passes is evidence of nothing. A test that FAILS when you break
# the behaviour is evidence it is wired to that behaviour. This repo has shipped
# multiple gates that passed vacuously - an ordinal guard the test never
# reached, a band assertion using the same params that produced the data, an
# NFC check comparing a value to its own idempotent normalization. Roughly one
# in eight tests written in a single session was initially worthless, and that
# only became visible when someone mutation-tested them.
#
# Usage:
#   ops/mutate.sh --all                        # every mutation in the registry
#   ops/mutate.sh <file> <old> <new> [pytest args...]
#
# Exit 0 when every mutation was KILLED. Exit 1 if any SURVIVED, or if a
# mutation could not be applied at all - a registry entry whose `old` string no
# longer exists is itself a silent failure, so it is an error, never a pass.
set -uo pipefail

cd "$(dirname "$0")/.."
REGISTRY="ops/mutations.toml"
FAILED=0

# Restoring must bump the mtime and drop cached bytecode. `mv` from a backup
# taken before the mutation restores an OLDER mtime than the .pyc written
# during the run, and a same-length mutation ("a" -> "w") leaves the size
# unchanged - so Python's (mtime, size) check keeps serving the MUTATED
# bytecode afterwards. That silently poisoned every later test run until it was
# traced back here (2026-07-28).
restore() {
    [[ -n "${BACKUP:-}" && -f "$BACKUP" ]] || return 0
    mv "$BACKUP" "$TARGET"
    touch "$TARGET"
    find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 2>/dev/null |
        xargs -0r rm -rf
}
trap 'restore' EXIT INT TERM

run_one() {
    TARGET="$1"; local old="$2" new="$3"; shift 3
    local selector=("$@")
    [[ ${#selector[@]} -eq 0 ]] && selector=("-q")

    if [[ ! -f "$TARGET" ]]; then
        echo "ERROR  $TARGET does not exist"; FAILED=1; return
    fi
    # Counted in python, not grep: a multi-line `old` cannot be matched with
    # grep -F, and an uncountable pattern must not read as "absent".
    local hits
    hits=$(COUNT_FILE="$TARGET" COUNT_PAT="$old" python3 -c 'import os,pathlib;print(pathlib.Path(os.environ["COUNT_FILE"]).read_text().count(os.environ["COUNT_PAT"]))')
    if [[ "$hits" -gt 1 ]]; then
        # Ambiguity silently mutates the FIRST match, which may not be the site
        # under test. That happened on the first real run: an `open("a", ...)`
        # pattern hit append_raw instead of append_receipt, and the surviving
        # mutation was blamed on the wrong gate.
        echo "ERROR  ambiguous in $TARGET ($hits matches) - add context: ${old:0:50}"
        FAILED=1; return
    fi
    if [[ "$hits" -eq 0 ]]; then
        # The registry has drifted from the code. Treat as failure: a mutation
        # that cannot be applied silently stops testing anything.
        echo "ERROR  cannot apply to $TARGET - not found: ${old:0:60}"
        FAILED=1; return
    fi

    BACKUP="$(mktemp)"; cp "$TARGET" "$BACKUP"
    python3 - "$TARGET" "$old" "$new" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace(sys.argv[2], sys.argv[3], 1))
PY
    if uv run pytest "${selector[@]}" >/dev/null 2>&1; then
        echo "SURVIVED  $TARGET  ${old:0:50}"
        echo "          the suite stayed green with this broken - the gate is vacuous"
        FAILED=1
    else
        echo "killed    $TARGET  ${old:0:50}"
    fi
    restore; BACKUP=""
}

if [[ "${1:-}" == "--all" ]]; then
    [[ -f "$REGISTRY" ]] || { echo "no $REGISTRY"; exit 1; }
    if ! n=$(python3 -c "import tomllib;print(len(tomllib.load(open('$REGISTRY','rb'))['mutation']))" 2>&1); then
        echo "ERROR  cannot parse $REGISTRY:"; echo "$n" | tail -2; exit 1
    fi
    # Zero mutations must never read as success. The first version of this
    # harness printed "OK - every mutation was killed" after the registry
    # failed to parse and the loop ran nothing - the exact vacuous pass it
    # exists to detect, in itself.
    [[ "$n" -gt 0 ]] || { echo "ERROR  $REGISTRY declares no mutations"; exit 1; }
    echo "running $n mutations from $REGISTRY"
    for i in $(seq 0 $((n - 1))); do
        # NUL-separated, not newline: `mapfile -t` splits on newlines, so a
        # multi-line `old` was silently truncated to its first line - which then
        # matched several sites and reported a unique entry as ambiguous.
        mapfile -d '' -t f < <(MUT_I="$i" MUT_REG="$REGISTRY" python3 -c '
import os, tomllib, sys
m = tomllib.load(open(os.environ["MUT_REG"], "rb"))["mutation"][int(os.environ["MUT_I"])]
sys.stdout.write("\0".join([m["file"], m["old"], m["new"], m.get("tests", "")]))
')
        if [[ -n "${f[3]}" ]]; then run_one "${f[0]}" "${f[1]}" "${f[2]}" "${f[3]}" "-q"
        else run_one "${f[0]}" "${f[1]}" "${f[2]}"; fi
    done
else
    [[ $# -ge 3 ]] || { sed -n '5,12p' "$0"; exit 1; }
    run_one "$@"
fi

[[ $FAILED -eq 0 ]] && echo "OK - every mutation was killed" || echo "FAIL - see SURVIVED/ERROR above"
exit $FAILED
