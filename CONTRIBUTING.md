# Contributing

phenoforge is developed as part of the Fragua research line. Until 1.0 the API may
move; issues and PRs are welcome once the repository is public.

Ground rules:

- Every model family MUST carry its primary reference (real citation; DOI where one
  exists). A family without a citation is not merged.
- Physical bounds, not numerical conveniences: parameter ranges come from the
  literature or from stated physical reasoning recorded in the docstring.
- Pure numpy/scipy in the core (Pyodide-safe); heavy dependencies only behind
  optional extras.
- Tests: every new family ships shape/limit/known-value tests; every new ensemble
  method ships simplex/ordering/recovery tests.
- Style: ruff (line length 100), English only, no em-dash, no emoji.

Workflow: branch from `develop` (`task/<topic>`), PR to `develop`, promotion to
`main` by the maintainer.
