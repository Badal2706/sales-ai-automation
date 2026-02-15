### 🚀 AI Sales Automation System (Local MVP)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Database](https://img.shields.io/badge/Database-SQLite-lightgrey)


A local AI-powered Sales Assistant built using Python, Streamlit, SQLite, and on-device LLMs to automate CRM structuring and follow-up generation from sales conversations.

This Phase-1 MVP focuses on fully local processing with modular production-style architecture.

### 🎯 Objective

To eliminate manual sales data handling by automatically:

• Converting sales conversations into structured CRM records
• Generating professional follow-up emails
• Creating short WhatsApp-style follow-up messages
• Managing multiple clients and interaction history
• Persisting everything in a relational database

All without using any cloud APIs.

### 🧠 Core Features

✅ Text-based conversation input
✅ Local LLM powered CRM structuring (strict JSON output)
✅ Automated follow-up email generation
✅ Automated short message generation
✅ Multi-client support
✅ Interaction history tracking
✅ SQLite relational storage
✅ Live Streamlit UI updates

### 🛠 Tech Stack

**Backend & Logic**

    Python
    SQLite (relational database)
    JSON validation & parsing

**AI & NLP**

    Local LLM via Ollama (or transformer-based local inference)
    Prompt engineering for structured data extraction
    Context-aware follow-up generation using client history

**Frontend**

    Streamlit multi-page application
    Real-time database-driven UI

**Architecture**

    Modular production-style codebase
    Separation of AI logic, database layer, UI, and prompts

### 📁 Project Structure

    sales_ai/
    │
    ├── app.py                 # Streamlit UI
    ├── database.py            # SQLite setup & CRUD operations
    ├── ai_crm.py              # CRM structuring AI logic
    ├── ai_followup.py         # Follow-up generation AI
    ├── memory.py              # Client history retrieval
    ├── models.py              # Database schema definitions
    ├── prompts.py             # AI prompt templates
    ├── config.py              # App configuration
    ├── requirements.txt
    └── sales_ai.db            # Local database (ignored in Git)


### 📊 Database Design

**clients table**

    id (PK)
    name
    company
    email
    created_at

**interactions table**

    id (PK)
    client_id (FK)
    date
    raw_text
    summary
    deal_stage
    objections
    interest_level
    next_action
    followup_date

**followups table**

    id (PK)
    interaction_id (FK)
    email_text
    message_text

### ⚙️ Installation

**1️⃣ Clone repository**

    gh repo clone Badal2706/sales-ai-automation
    cd sales-ai-automation

**2️⃣ Create virtual environment**

    python -m venv .venv
    .venv\Scripts\activate     # Windows
    source .venv/bin/activate # Mac/Linux

**3️⃣ Install dependencies**

    pip install -r requirements.txt

**▶️ Run Application**

    streamlit run app.py

### 🧪 How It Works

1. Select existing client or create new
2. Paste sales conversation text
3. Local AI structures CRM data in JSON
4. AI generates follow-up email and message
5. Data saved in SQLite
6. UI updates instantly

### 🚧 Phase-2 Roadmap

- Audio call transcription
- Vector-based retrieval memory
- Advanced lead analytics dashboard
- Multi-user authentication
-Automated email sending


### Built with ❤️ by Badal Patel

