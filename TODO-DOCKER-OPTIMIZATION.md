# Docker Optimization & GPU Embedding - Implementation Plan

## Overview
Optimize Docker build process using uv + multi-stage builds, and move Embedding API to GPU.

## Architecture Change
- **Before**: 3 separate containers (embed-api CPU, rerank-api GPU, backend)
- **After**: 2 containers (ml-api GPU unified, backend)

---

## Phase 1: Create Unified ML API Service

### TODO 1.1: Create ml-api directory structure
- [ ] Create `ml-api/` directory
- [ ] Create `ml-api/pyproject.toml` with uv + CUDA 13.0 PyTorch
- [ ] Create `ml-api/ml_api.py` with unified /embed and /rerank endpoints
- [ ] Create `ml-api/Dockerfile` with multi-stage build

### TODO 1.2: Implement unified ml_api.py
- [ ] Merge embed_api.py logic (GPU ONNX)
- [ ] Merge rerank_api.py logic (GPU FlagReranker)
- [ ] Single FastAPI app on port 8001
- [ ] Path-based routing: /embed, /rerank, /health
- [ ] Both models load at startup ("hot" API)

### TODO 1.3: Generate uv.lock
- [ ] Run `uv lock` to generate lockfile

### TODO 1.4: Test ML API locally
- [ ] Test /embed endpoint
- [ ] Test /rerank endpoint
- [ ] Test /health endpoint
- [ ] Verify GPU usage for both models

---

## Phase 2: Optimize Backend Dockerfile

### TODO 2.1: Create uv-based backend Dockerfile
- [ ] Multi-stage build with uv
- [ ] Layer caching for dependencies
- [ ] Minimal runtime image

---

## Phase 3: Update docker-compose.yml

### TODO 3.1: Update compose configuration
- [ ] Remove embed-api service
- [ ] Remove rerank-api service
- [ ] Add ml-api service with GPU reservation
- [ ] Update backend environment variables (single EMBED_API_URL)

---

## Phase 4: Refactor Backend for Single ML API Endpoint

### TODO 4.1: Update backend embed client
- [ ] Update EMBED_API_URL to point to ml-api:8001
- [ ] Verify /embed calls work

### TODO 4.2: Update backend rerank client
- [ ] Update RERANK_API_URL to point to ml-api:8001
- [ ] Verify /rerank calls work

---

## Phase 5: Fix .gitignore

### TODO 5.1: Create comprehensive .gitignore
- [ ] Python artifacts
- [ ] Virtual environments
- [ ] IDE files
- [ ] OS files
- [ ] Docker files
- [ ] Environment files
- [ ] Model caches
- [ ] Test artifacts

---

## Phase 6: Cleanup

### TODO 6.1: Remove old services
- [ ] Delete embed-api/ directory
- [ ] Delete rerank-api/ directory

### TODO 6.2: Final testing
- [ ] Build all containers with docker compose build
- [ ] Run docker compose up
- [ ] Test full pipeline (upload, query, rerank)

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | ✅ Done | ml-api created with pyproject.toml, ml_api.py, Dockerfile, uv.lock |
| Phase 2 | ✅ Done | Backend Dockerfile optimized with uv multi-stage build |
| Phase 3 | ✅ Done | docker-compose.yml updated, unified ml-api service |
| Phase 4 | 🔲 Pending | Backend refactor needed for single ML API |
| Phase 5 | ✅ Done | Comprehensive .gitignore created |
| Phase 6 | 🔲 Pending | Cleanup + final testing |

---

## Technical Notes

### CUDA 13.0 PyTorch Index
```
https://download.pytorch.org/whl/cu130
```

### ONNX Runtime GPU
Use `onnxruntime-gpu` instead of `onnxruntime` for GPU inference.

### Port Configuration
- ml-api: 8001 (single port, path-based routing)
- backend: 8000
- qdrant: 6333, 6334
- frontend: 3000
