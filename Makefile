SHELL := /bin/bash

.PHONY: bootstrap lint test docker-build docker-up docker-smoke down clean

bootstrap:
	cd backend && uv sync --all-extras --dev
	cd ml-api && uv sync --dev
	cd frontend && pnpm install --frozen-lockfile

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd ml-api && uv run ruff check . && uv run ruff format --check .
	cd frontend && pnpm run lint

test:
	cd backend && uv run pytest --cov=app --cov-report=term-missing --cov-report=xml --cov-fail-under=63
	cd ml-api && ML_API_READY_TIMEOUT=5 uv run pytest --cov=cache_config --cov-report=term-missing --cov-report=xml --cov-fail-under=90

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-smoke:
	docker compose up -d
	@for url in http://localhost:8001/health http://localhost:8000/api/health http://localhost:3000/ ; do \
		echo "Checking $$url"; \
		for i in $$(seq 1 60); do \
			if curl -fsS "$$url" >/dev/null; then \
				break; \
			fi; \
			if [ "$$i" -eq 60 ]; then \
				echo "Smoke check failed for $$url"; \
				exit 1; \
			fi; \
			sleep 5; \
		done; \
	done

down:
	docker compose down

clean:
	docker compose down -v --remove-orphans
