# Contributing to Document-RAG

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

-   Reporting a bug
-   Discussing the current state of the code
-   Submitting a fix
-   Proposing new features
-   Becoming a maintainer

## Development Process

We use Github to host code, to track issues and feature requests, and to accept pull requests.

1.  **Fork the repo** and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes.
5.  Make sure your code lints.
6.  Issue that pull request!

## Local Development (Quick Reference)

### Backend
-   Uses `uv` for dependency management.
-   Run tests: `cd backend && uv run pytest`
-   Linting: `cd backend && uv run ruff check .`

### Frontend
-   Uses `pnpm`.
-   Run dev server: `cd frontend && pnpm run dev`
-   Linting: `cd frontend && pnpm run lint`

### Data
-   Qdrant runs in Docker: `docker-compose up -d qdrant`

## Pre-commit Hooks

We use pre-commit hooks to ensure consistency.
To install:
```bash
cd backend
uv run pre-commit install
```

## Pull Request Process

1.  Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
2.  Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
3.  The CI pipeline must pass before merging.

## Any questions?

Feel free to open an issue or start a discussion.
