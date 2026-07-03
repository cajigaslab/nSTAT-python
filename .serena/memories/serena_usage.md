# Serena MCP — usage rules & gotchas

Generic guidance for using Serena's semantic (LSP-backed) code tools well on this repo. Not repo-specific.

## Core discipline
- **Prefer Serena's semantic tools over grep/Read for CODE.** Agents drift back to grep/Read as context
  grows (a documented Serena problem); resist it. Serena `find_symbol`/`get_symbols_overview` are more
  token-efficient and precise. Use plain Read/Edit only for NON-code files (Markdown/YAML/JSON) or a file
  you've already fully read.
- **Read loop:** `get_symbols_overview(file)` (the map) → `find_symbol("Class", depth=1)` (methods, no
  bodies) → `find_symbol("Class/method", include_body=True)` (only the body you need) →
  `find_referencing_symbols(...)` before changing a signature.
- **Edit loop:** read the symbol first (`include_body=True`), then `replace_symbol_body` /
  `insert_before/after_symbol` / `replace_content(regex ...)`, then `get_diagnostics_for_file` (a run-free
  error check).
- **Serena line numbers are 0-based** (unlike most tools).

## Name-path syntax (how symbols are addressed)
- `Name` — suffix match (any symbol so named). `Class/method` — a member. `/Class/method` — absolute
  (exact full path). `Class/method[1]` — a specific overload. `substring_matching=True` matches the last
  segment as a substring.
- Every symbol tool takes `name_path`/`name_path_pattern` + a `relative_path` (scope to a file/dir).

## Tool map
- **Read:** get_symbols_overview, find_symbol, find_referencing_symbols, find_declaration,
  find_implementations, get_diagnostics_for_file, search_for_pattern, list_dir, find_file.
- **Edit:** replace_symbol_body, insert_before_symbol, insert_after_symbol, replace_content (in-file
  regex/literal), replace_in_files (MULTI-FILE regex with dry-run + per-occurrence selection),
  rename_symbol, safe_delete_symbol, create_text_file.
- **Memory:** list/read/write/edit/rename/delete_memory (`rename_memory` updates `mem:` refs).
- **Session/config:** initial_instructions, onboarding, activate_project, get_current_config,
  open_dashboard, restart_language_server.

## Gotchas (learned + doc-verified)
- **WORKTREE EDIT HAZARD (critical).** Serena EDIT tools act on the ACTIVE PROJECT ROOT — usually the main
  repo, NOT a git worktree. An agent editing in a worktree via `replace_symbol_body` etc. silently writes
  to the main repo. Fix: `activate_project("<worktree path>")` first (confirm with `get_current_config`),
  OR use plain Read/Edit/Write in the worktree. Serena READ tools are always safe.
- **`rename_symbol` is LSP-only and often incomplete.** It renames the symbol definition + code references
  the language server resolves, but MISSES string literals (imports written as strings, `__all__` entries,
  docstrings) and sometimes cross-file refs. Always follow a rename with `replace_in_files` for the string
  refs and a `grep` to confirm zero of the old name remain.
- **`find_declaration`/`find_implementations` are language-server-limited** — they generally do NOT resolve
  into external dependencies, and implementations are only partial for some languages.
- **Schemas are deferred.** Serena tools appear by name but their schema must be loaded via `ToolSearch`
  (`select:mcp__serena__<tool>`) before the first call.
- **Multiple instances = multiple dashboard ports.** If several Serena servers run (e.g. Claude Desktop +
  Claude Code), each dashboard is on an auto-incremented port (24282, 24283, …). `get_current_config` / the
  launch log shows the active project + port; the dashboard shows live tool calls + logs.
- **Edit granularity:** `replace_symbol_body` = a whole function/class (read it first). `replace_content`
  regex = a few lines inside a symbol (use wildcards `start.*?end`, don't re-quote the whole span).
  `insert_*_symbol` = a new symbol/import. Non-code file → plain Read/Edit.

## Environment best practices (Serena docs)
- Work from a CLEAN git state so the model can `git diff` to self-correct; start with tests + lint passing
  (Serena has no debugger — it relies on run/lint/test output).
- Type annotations help the language server on dynamically-typed code.
- Do NOT install Serena via an MCP/plugin marketplace (outdated commands); use `uv tool install serena-agent`.

## Config location
Everything lives under `.serena/`: `project.yml` (versioned), `.serena/.gitignore` (ignores `/cache` +
`/project.local.yml` — machine-local), and `memories/*.md` (versioned — the shared project knowledge).
Commit `project.yml` + `.gitignore` + `memories/`; never commit `cache/`.
