# DLCV (Deep Learning Computer Vision - Video Redaction)

Full-stack application for automated video redaction, object detection, tracking, Re-ID, and ghost box filtering.

## Architecture

- **Frontend (`/src`)**: React 19, TypeScript, Vite, Tailwind CSS v4, Zustand, TanStack Query, Oxlint
- **Backend (`/backend`)**: FastAPI, Python 3.11+, Pydantic v2, WebSockets, OpenCV/PyTorch video processing pipeline

## Common Commands

### Frontend
- **Dev Server**: `npm run dev` (starts Vite on default port)
- **Build**: `npm run build` (`tsc -b && vite build`)
- **Lint**: `npm run lint` (`oxlint`)
- **Preview**: `npm run preview`
*(Note: No unit/component test runner such as Vitest or Jest is currently installed in `package.json`)*

### Backend
- **Virtual Environment**: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (POSIX)
- **Install Dependencies**: `pip install -r backend/requirements.txt`
- **Run Server**: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
- **Pipeline / Integration Smoke Test**: `python test_script.py` (requires running backend server)
*(Note: No unit test runner such as pytest is currently installed in `backend/requirements.txt`)*

## Code Conventions & Standards

- **TypeScript/React**: Follow React 19 functional component conventions. Use strict types (avoid `any`). Adhere to Oxlint rules.
- **Python/FastAPI**: Follow PEP 8 and FastAPI best practices. Use Pydantic schemas for request/response contracts and dependency injection for services.
- **Error Handling**: Maintain explicit domain exceptions and status codes in `backend/core/exceptions.py`.
