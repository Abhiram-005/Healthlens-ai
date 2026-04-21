<<<<<<< HEAD
# 🩺 HealthLens AI
### Cognizant Technoverse Hackathon 2026 — Healthcare · Clinical Documentation

AI-powered medical document analysis platform. Upload any medical document
(lab report, prescription, discharge summary) and get instant plain-English insights,
specialist recommendations, and a health timeline — with built-in hallucination protection.

---

## Architecture

```
Upload → OCR Node → Agent 1 (Extract) → Comparison Node → Agent 3 (Verify) → Agent 2 (Recommend)
```

| Component | Role | Technology |
|---|---|---|
| OCR Node | Convert PDF/image to text | Tesseract + pdfplumber + OpenCV |
| Agent 1 | Extract ALL medical entities from OCR text | Groq LLM (llama-3.3-70b-versatile) |
| Comparison Node | Compare with previous report | Deterministic Python logic |
| Agent 3 | Verify every extracted value against OCR | Groq LLM + rule-based checks |
| Agent 2 | Plain-language insights + specialist recs | Groq LLM (llama-3.3-70b-versatile) |
| Chat | Grounded Q&A about the report | Groq LLM (llama-3.1-8b-instant) |
| Backend | REST API server | FastAPI (Python) |
| Frontend | Full dashboard UI | Vanilla HTML/CSS/JS |
| Database | Report storage & history | SQLite |

---

## 🆓 Free APIs Used

| Service | What For | Cost |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference (Llama 3.3 70B) | **FREE** — 30 req/min |
| Tesseract OCR | Image/scanned PDF text extraction | **FREE** — open source |
| pdfplumber | Digital PDF text extraction | **FREE** — open source |
| OpenCV | Image preprocessing (noise removal, deskew) | **FREE** — open source |

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- **Python 3.10+**  (check: `python3 --version`)
- **pip** (check: `pip --version`)
- **Tesseract OCR** (system package — see below)
- **Poppler** (needed by pdf2image for PDF → image conversion)

### Install Tesseract + Poppler

**Ubuntu / Debian / WSL:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
```

**macOS (Homebrew):**
```bash
brew install tesseract poppler
```

**Windows:**
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH (e.g. `C:\Program Files\Tesseract-OCR`)
3. Download Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
4. Add Poppler `bin/` folder to PATH

---

## 🚀 Step-by-Step Setup & Run

### Step 1 — Clone / Unzip the project

If you received this as a zip:
```bash
unzip healthlens-ai.zip
cd healthlens-ai
```

### Step 2 — Get a FREE Groq API Key

1. Go to: **https://console.groq.com**
2. Sign up (free, no credit card required)
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with `gsk_...`)

### Step 3 — Set up Environment Variables

```bash
cd healthlens-ai
cp .env.example .env
```

Open `.env` in any text editor and replace `your_groq_api_key_here` with your actual key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 4 — Create Python Virtual Environment

```bash
python3 -m venv venv
```

Activate it:
```bash
# Linux / macOS / WSL:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should now see `(venv)` at the start of your terminal prompt.

### Step 5 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, Groq SDK, Tesseract wrapper, OpenCV, pdfplumber, and all other dependencies.
It may take 2–4 minutes.

### Step 6 — Run the Server

```bash
cd backend
python main.py
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
HealthLens AI starting...
Model (accurate): llama-3.3-70b-versatile
Model (fast):     llama-3.1-8b-instant
```

### Step 7 — Open the App

Open your browser and go to: **http://localhost:8000**

You should see the HealthLens AI dashboard. The status indicator in the top bar will show 
"Groq Connected" in green if your API key is working.

---

## 🧪 Testing the App

1. **Upload a medical document** — PDF, JPG, PNG, or any image of a lab report
2. Watch the **pipeline steps** animate in real time
3. After analysis, explore:
   - **Test Insights** — color-coded cards for each test result
   - **All Tests** — full table with reference ranges and confidence scores
   - **Specialists** — who to consult and with what urgency
   - **Timeline** — comparison with a previous report (upload same patient twice)
   - **Next Steps** — practical action items
   - **Raw OCR** — see exactly what text was extracted
4. Use the **chat box** to ask follow-up questions about the report

---

## 📂 Project Structure

```
healthlens-ai/
├── backend/
│   ├── main.py              ← FastAPI server (entry point)
│   ├── pipeline.py          ← Pipeline orchestrator
│   ├── ocr_processor.py     ← OCR workflow node
│   ├── agent1_extractor.py  ← Agent 1: Medical entity extraction
│   ├── agent2_recommender.py← Agent 2: Insights + chat
│   ├── agent3_verifier.py   ← Agent 3: Hallucination guard
│   ├── database.py          ← SQLite storage
│   └── models.py            ← Pydantic data models
├── frontend/
│   └── index.html           ← Full dashboard (single file)
├── uploads/                 ← Uploaded files (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠 Common Issues

### "GROQ_API_KEY is not set"
- Make sure you copied `.env.example` to `.env` (not just edited the example)
- Make sure your key starts with `gsk_`

### "tesseract is not installed or not in PATH"
- Ubuntu: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH

### "pdf2image: Unable to get page count. Is poppler installed?"
- Ubuntu: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`
- Windows: download poppler and add `bin/` to PATH

### "ModuleNotFoundError: No module named 'cv2'"
- Run: `pip install opencv-python-headless`

### Groq Rate Limit Error
- The free Groq tier allows 30 requests/min and 6000 tokens/min
- For large documents, wait a few seconds between uploads
- Or upgrade your Groq plan (still very affordable)

### Port already in use
- Change the port: edit `.env` → `PORT=8080` and restart
- Or kill the existing process: `lsof -ti:8000 | xargs kill`

---

## 🔒 Accuracy & Safety Features

- **Zero temperature** LLM calls on extraction and verification — maximum determinism
- **source_text verification** — every extracted value must be traceable to original OCR text
- **Dual verification** — rule-based check + LLM cross-check
- **Confidence scoring** — every field gets HIGH/MEDIUM/LOW confidence
- **Unsupported claims removal** — Agent 3 removes anything not found in source text
- **Built-in medical reference ranges** — 20+ common tests validated against known ranges
- **Mandatory disclaimer** — every response includes a medical disclaimer

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Server health + Groq status |
| POST | `/api/analyze` | Upload & analyze document |
| POST | `/api/chat` | Ask a question about a report |
| GET | `/api/history` | Get list of all reports |
| GET | `/api/report/{id}` | Get a specific report |
| DELETE | `/api/report/{id}` | Delete a report |

---

## 🏆 Cognizant Technoverse 2026

This project is submitted under the **Healthcare — Clinical Documentation** theme.

Tech stack used:
- **Frontend:** Vanilla HTML/CSS/JS (React-compatible output format)
- **Backend:** FastAPI (Python)
- **AI/ML:** LangChain-style pipeline using Groq (Llama 3.3 70B) + LangGraph-inspired multi-agent flow
- **OCR:** AWS Textract-compatible design (uses Tesseract locally, swappable to Textract)
- **Databases:** SQLite (prototype) → PostgreSQL + MongoDB (production)
- **Cloud:** AWS-compatible architecture (containerised FastAPI)

---

*© 2026–2027 Cognizant Technoverse | HealthLens AI Team*
=======
# healthlens-ai
>>>>>>> f4421f3d5ba74269476f39dffda85c8592580d08
