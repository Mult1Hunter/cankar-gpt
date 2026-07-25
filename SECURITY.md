# Security Policy

CankarGPT is a personal, educational ML project. There is no production service
behind this repository (yet) - but security reports are welcome at any stage.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** on this repository
(Security tab -> "Report a vulnerability"). Please do not open public issues for
security-sensitive findings.

You can expect an acknowledgement within a few days. There is no bug bounty.

## Scope notes

- No secrets live in this repository by policy (enforced by gitleaks pre-commit
  hooks and CI). If you believe you have found one in the history, report it
  privately as above.
- Model weights and datasets are distributed via Hugging Face Hub, not this repo.
- The demo services (ROADMAP Phases 7-8) will get their own scope notes when
  they exist.
