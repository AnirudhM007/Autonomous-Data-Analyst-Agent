# Contributing to Autonomous Data Analyst Agent

Thank you for your interest in contributing to **Autonomous Data Analyst Agent**! We welcome bug fixes, documentation improvements, new analysis tools, and UI enhancements.

---

## 🛠️ Development Environment Setup

### Prerequisites
- **Python**: 3.11 or newer
- **Node.js**: 20.x or newer
- **npm** or **pnpm** (10+)
- **Git**

### 1. Clone & Fork the Repository
```bash
git clone https://github.com/AnirudhM007/Autonomous-Data-Analyst-Agent.git
cd Autonomous-Data-Analyst-Agent
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Start the backend API server:
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Frontend Setup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧪 Testing & Verification

Every pull request must pass the automated test suite and type check:

### Run Backend Tests
```bash
cd backend
python -m pytest tests -v
```

### Run Frontend Typecheck & Build
```bash
cd frontend
npm run build
```

---

## 📐 Architecture & Contribution Rules

To preserve the deterministic security model and prevent LLM hallucinations:
1. **Zero Raw Data to Models**: Never transmit raw CSV row cells or sensitive table values to LLM APIs. Only dataset profile metadata (column names, inferred datatypes, row counts) may be passed for structured planning.
2. **Deterministic Tool Registration**: All mathematical, statistical, aggregation, and outlier logic must reside in Python functions within `backend/app/services/tools.py`. LLMs must **never** execute arbitrary Python, SQL, or shell commands.
3. **Strict Parameter Validation**: Every parameter passed into an analysis tool must be validated against `profile["column_names"]` and appropriate data types.
4. **Isolated Process Execution**: Long-running or heavy tabular operations must execute within child processes subject to the 90-second execution deadline.
5. **No Hallucinated Claims**: All findings displayed in reports must directly cite verifiable numbers and segment boundaries computed by the registered tools.

---

## 🌿 Git Workflow & Branch Naming

1. Create a descriptive branch from `main`:
   - `feat/new-analysis-tool` (New analytical capabilities)
   - `fix/date-parsing-edge-case` (Bug fixes)
   - `docs/clarify-docker-deployment` (Documentation updates)
   - `refactor/sse-event-handler` (Code restructuring)
2. Commit with concise, descriptive commit messages adhering to Conventional Commits:
   - `feat: add mutual information ranker to registered tools`
   - `fix: handle empty string column names during csv ingestion`
   - `docs: update system architecture diagram and sse spec`
3. Push to your fork and submit a Pull Request to `main`.

---

## ✅ Pull Request Checklist

Before submitting a PR, ensure:
- [ ] Pytest passes all 15 tests cleanly without failures (`pytest tests`).
- [ ] Frontend builds without TypeScript or bundling errors (`npm run build`).
- [ ] No API keys, credentials, or `.env` files are tracked in git.
- [ ] New registered tools are documented in both `backend/app/services/tools.py` and `ARCHITECTURE.md`.
- [ ] Changes adhere to the Retro Neubrutalism design system for frontend contributions.

---

## 📜 Code of Conduct

Please maintain a collaborative, respectful, and constructive environment for all contributors and maintainers.
