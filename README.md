# 🚀 Corporate Finance Autopilot — Backend

**Corporate Finance Autopilot** is an agentic AI platform designed to automate complex financial analysis. It combines real-time data ingestion, sophisticated financial modeling, and Retrieval-Augmented Generation (RAG) to provide institutional-grade insights.

**🔗 Live Application:** [corporate-finance-autopilot-fronten.vercel.app](https://corporate-finance-autopilot-fronten.vercel.app/)

---

## 🌟 Key Features

* **Agentic AI Loop:** A self-reasoning agent powered by **Llama-3.3-70B** that autonomously calls tools, searches filings, and builds models.
* **Real-Time Data Ingestion:** Architecture ensuring fresh data from **Yahoo Finance**, **FRED Macro API**, and **SEC EDGAR**.
* **Advanced Financial Engine:** Automates 3-scenario $DCF$ valuations, $CAPM$-based $WACC$ calculations.
* **High-Performance RAG:** Semantic search over SEC filings and financials using **Neon.AI (pgvector)** and offline **HuggingFace** embeddings.
* **Automated Reporting:** Generates structured Investment Memos and Equity Research notes in downloadable **PDF** format.
* **WebSocket Streaming:** Real-time execution traces allow users to see the AI's "thought process" as it works.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework** | FastAPI (Python 3.11+)  |
| **LLM Orchestration** | Together AI (Llama 3.3 70B / Mistral 7B)  |
| **Vector Database** | Neon.AI (PostgreSQL + `pgvector`)  |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (Offline) |
| **Data Sources** | `yfinance`, FRED API, SEC EDGAR  |
| **Deployment** | Docker, Google Cloud  |

---

## 📂 Project Structure

```text
backend/
├── app/
│   ├── main.py              # Application entrypoint & RAG init 
│   ├── models/              # Pydantic data schemas 
│   ├── routers/             # API Endpoints (Agent, Analysis, Company, Reports) 
│   ├── services/
│   │   ├── agent/           # Agentic loop & Tool definitions
│   │   ├── financial_engine/# DCF & WACC logic
│   │   ├── ingestion/       # External API integrations (YF, FRED, SEC)
│   │   ├── rag/             # Vector storage & Semantic search
│   │   └── report/          # PDF Generation (fpdf2)
├── Dockerfile               # Container build file 
└── requirements.txt         # Project dependencies
