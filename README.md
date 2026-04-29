# 🩺 HealthLens AI

### AI-Powered Medical Document Intelligence System

HealthLens AI is an intelligent healthcare platform that analyzes medical documents and transforms them into **structured, easy-to-understand insights** using advanced AI techniques.

It helps users interpret complex reports, track health changes, and receive AI-assisted recommendations — all in one place.

---

## 🚀 Overview

HealthLens AI processes medical documents through a **multi-stage AI pipeline**, ensuring both **accuracy and reliability**.

From raw PDFs or images → to verified insights → to meaningful recommendations.

---

## ✨ Core Features

* 🧾 **Medical Document Processing**
  Upload prescriptions, lab reports, or discharge summaries

* 🔍 **OCR-Based Text Extraction**
  Converts scanned documents into machine-readable text

* 🤖 **Multi-Agent AI System**
  Specialized agents handle extraction, verification, and recommendations

* 🛡️ **Hallucination Detection Layer**
  Ensures AI outputs are validated and trustworthy

* 📊 **Health Report Comparison**
  Tracks variations across multiple reports

* 💬 **Context-Aware Chat System**
  Ask questions based on your uploaded medical data

* ⚡ **Fast Processing Pipeline**
  Optimized for real-time or near real-time responses

---

## 🧠 System Architecture

```text
Input Document
      ↓
OCR Processing (Tesseract + OpenCV)
      ↓
Medical Data Extraction (LLM Agent)
      ↓
Report Comparison Engine
      ↓
Verification Agent (Anti-Hallucination)
      ↓
Insight & Recommendation Agent
      ↓
User Dashboard / Chat Interface
```

---

## 🛠️ Tech Stack

### 🔹 Backend

* **Python**
* **FastAPI** (API framework)

### 🔹 Frontend

* **HTML, CSS, JavaScript**

### 🔹 AI & LLM

* **Groq API (LLaMA-based models)**
* Prompt-engineered multi-agent workflow

### 🔹 OCR & Document Processing

* **Tesseract OCR**
* **OpenCV**
* **pdfplumber**

### 🔹 Database

* **SQLite**

### 🔹 Other Tools

* **dotenv** (environment management)
* **Uvicorn** (ASGI server)

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Abhiram-005/Healthlens-ai.git
cd Healthlens-ai
```

### 2. Setup Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 3. Install Dependencies

```bash
python -m venv venv

# Activate environment
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

### 4. Run Application

```bash
cd backend
python main.py
```

Open:
http://localhost:8000

---

## 🎯 Applications

* Medical report understanding
* Patient health monitoring
* AI-assisted healthcare insights
* Educational healthcare tools

---

## ⚠️ Disclaimer

This system is intended for **educational and research purposes only**.
It does **not provide medical diagnosis or professional healthcare advice**.

---

## 🏆 What Makes This Project Stand Out

* Multi-agent AI architecture instead of single-model output
* Built-in hallucination verification layer
* End-to-end pipeline from document → insight
* Real-world healthcare application use case

---
