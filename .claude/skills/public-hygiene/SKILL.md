---
name: public-hygiene
description: Paranoid pre-push pass for this public repo - full-history secrets scan, private-info grep, tracked-file and size audit. Run before any push or tag push.
---

# Public-hygiene pass

This repo is public from commit #1 - anything pushed is permanently public
(forks and caches survive force-pushes). Run this BEFORE `git push`, never after.

## Checks (run all, report all findings before concluding)

1. **Secrets, full history:**
   `gitleaks detect --redact` from the repo root - must be clean.
2. **Tracked-file audit:**
   - `git ls-files | grep -E '(^|/)\.env$|\.(pem|key)$'` -> must be empty
   - `git ls-files data/ checkpoints/ | grep -v 'README\.md$'` -> must be empty
     (data/README.md is the governed-dir contract - ADR 0007, not a dataset)
3. **Private-info grep over full history:**
   `git log --all -p | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/home/[a-z]+|ssh://|BEGIN (RSA|OPENSSH|EC) PRIVATE'`
   Review every hit: real IPs, private paths, host aliases, VPS details are leaks.
   (Expected and fine: the author's public email, GitHub handle, and website
   `nextgen-solutions.xyz`; `0.0.0.0`/`127.0.0.1`.)
4. **Size audit (nothing heavy slipped in):**
   `git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | awk '$1=="blob" && $2>1000000'`
   -> expect no output.
5. **Commit-message eyeball:** `git log --all --oneline` - no client names, hostnames,
   or notes that belong in the private meta repo.
6. **Name consistency:** `git grep -inE 'cankar-?g[t]p'` -> must be empty.
   The project is CankarGPT / cankar-gpt everywhere (ADR 0002, amended 2026-07-25:
   the original slug transposed the final two letters of GPT); this drift class was
   caught once already, pre-commit-#1, and the guard now targets the old transposed
   spelling. (The `[t]` keeps this line from matching itself.)

## On any hit

- **Not yet pushed:** rewrite local history (fresh branch or rebase), then re-run
  this entire pass from step 1.
- **Already pushed:** treat any secret as burned - rotate it immediately; clean
  history only after rotation. Private info that isn't a credential: remove going
  forward, note it in the meta repo.
