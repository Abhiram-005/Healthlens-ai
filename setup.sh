#!/bin/bash
# HealthLens AI — Quick Setup Script
# Run: bash setup.sh
# Works on: Ubuntu / Debian / macOS / WSL

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   HealthLens AI — Setup Script           ║"
echo "║   Cognizant Technoverse 2026             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Detect OS ──────────────────────────────────────────────
OS="$(uname -s)"
echo "Detected OS: $OS"

# ── 2. Install system dependencies ───────────────────────────
echo ""
echo "▶ Installing Tesseract OCR and Poppler..."
if [[ "$OS" == "Linux" ]]; then
    sudo apt-get update -q
    sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils libgl1
elif [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Install from https://brew.sh"
        exit 1
    fi
    brew install tesseract poppler
else
    echo "Windows detected — please manually install Tesseract and Poppler."
    echo "See README.md for instructions."
fi

# ── 3. Check Python ───────────────────────────────────────────
echo ""
echo "▶ Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi
python3 --version

# ── 4. Create virtual environment ────────────────────────────
echo ""
echo "▶ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── 5. Install Python packages ────────────────────────────────
echo ""
echo "▶ Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

# ── 6. Copy .env ──────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  ACTION REQUIRED: Add your Groq API key         ║"
    echo "║                                                  ║"
    echo "║  1. Open .env in a text editor                  ║"
    echo "║  2. Replace 'your_groq_api_key_here'            ║"
    echo "║     with your key from https://console.groq.com ║"
    echo "║  3. Then run: source venv/bin/activate          ║"
    echo "║               cd backend && python main.py      ║"
    echo "╚══════════════════════════════════════════════════╝"
else
    echo ".env already exists — skipping copy."
fi

# ── 7. Create uploads dir ─────────────────────────────────────
mkdir -p uploads

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set GROQ_API_KEY"
echo "  2. Run:  source venv/bin/activate"
echo "  3. Run:  cd backend && python main.py"
echo "  4. Open: http://localhost:8000"
echo ""
