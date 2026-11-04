#!/bin/bash
# Release hygiene gates over the tracked files. Exits non-zero and lists every hit on failure.
# Usage: bash scripts/check_hygiene.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0
gate() {  # name, then the hits on stdin
  local hits
  hits=$(cat)
  if [ -n "$hits" ]; then echo "FAIL $1"; echo "$hits" | sed 's/^/  /'; fail=1; else echo "ok   $1"; fi
}

# Machine, user and site paths. Shebang lines are allowed; expanduser of a user-given path is allowed;
# the public repository's own URL is allowed.
git grep -nIiE '/tmp\b|/usr/(share|bin)/|mv_sam3|spark_camp|spark_main_mirror|LIBERO_upstream|bryceag11|biqw|cis261638|bgrant1|aism|/home/|/scratch|/anvil|/projects/|~/|Path\.home|import modal|modal\.(com|run)|\bcwru\b|\bcheng\b|\bomen\b|192\.168\.' \
  -- ':!scripts/check_hygiene.sh' | grep -vE '^[^:]+:1:#!' \
  | grep -vE 'github.com/CWRU-AISM/sparkbot|cwru-aism.github.io/spark-page' | gate "hardcoded paths and site names"

# Decorative banners: separator prints and comment rules.
git grep -nIE "print\(.*['\"][=-]{3,}|['\"][=-]['\"] *\* *[0-9]|^\s*(#|//|<!--|/\*) *[-=#*]{4,}|^\s*(#|//) *[-=]{2,} .*[-=]{4,}\s*$" \
  | gate "banners and separator prints"

# Files over 600 lines. Tests, the frozen prompt text, the repro item lists, uv.lock and assets/ are exempt.
git ls-files | grep -vE '(^|/)tests/|^src/spark_core/prompts/.*\.txt$|^repro/.*\.txt$|^uv\.lock$|^assets/' \
  | while read -r f; do [ -f "$f" ] && n=$(wc -l < "$f") && [ "$n" -gt 600 ] && echo "$f: $n lines"; done \
  | gate "files over 600 lines"

# Secrets: key literals and key or credential files.
{ git grep -nIE 'AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY'
  git ls-files | grep -iE '(^|/)(\.env|[^/]*api_key[^/]*|[^/]*secret[^/]*|credentials[^/]*|id_(rsa|ed25519)[^/]*|[^/]*\.(pem|key|p12))$'
} | gate "secrets and key files"

# ASCII only in tracked text files.
git grep -nIP '[^\x00-\x7F]' | gate "non-ASCII characters"

exit $fail
