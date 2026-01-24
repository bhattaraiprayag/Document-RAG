# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-   Implemented strict CI/CD pipeline using Github Actions.
-   Added `pre-commit` hooks for code hygiene and security (leaked secrets).
-   Added `CONTRIBUTING.md` and `LICENSE`.

### Changed
-   Migrated Frontend package manager from valid `npm` to `pnpm` for faster builds and disk space saving.
-   Migrated Backend linting from `flake8`/`black` to using `ruff` strictly.
-   Enforced multi-stage Docker builds.
-   Updated documentation to reflect new tooling (`uv`, `pnpm`).

### Fixed
-   Standardized dependency management across the repo.

## [0.1.0] - 2024-01-20
### Initial Release
-   Basic RAG functionality.
-   FastAPI Backend.
-   React Frontend.
-   Qdrant Integration.
