# IntelliData AI

An AI-powered data intelligence platform that automates dataset profiling,
quality assessment, intelligent cleaning, exploratory data analysis, and
generates AI-driven business insights and ML recommendations — with a
conversational chatbot for querying the dataset directly.

> Status: 🚧 Phase 1 — project scaffold. Feature pages land in later phases.

## Tech stack

Python · Streamlit · Pandas · scikit-learn · Plotly · Google Gemini API

## Local setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd IntelliData-AI

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and add your GEMINI_API_KEY

# 5. Run the app
streamlit run app.py
```

## Project structure

```
IntelliData-AI/
├── app.py              # Streamlit entry point
├── pages/              # feature pages (auto-discovered by Streamlit)
├── modules/             # data processing logic
│   └── ai/               # Gemini integration
├── utils/                # config, logging
├── uploads/ outputs/ reports/ logs/   # runtime data (git-ignored)
├── requirements.txt
├── .env.example
└── tests/
```

## Roadmap

- [x] Phase 1 — Project setup, config, logging
- [x] Phase 2 — Dataset upload & validation
- [x] Phase 3 — Data profiling
- [x] Phase 4 — Data quality scoring
- [x] Phase 5 — Intelligent cleaning engine
- [x] Phase 6 — Exploratory data analysis
- [x] Phase 7 — Visualization dashboard
- [x] Phase 8 — Gemini AI service layer
- [x] Phase 9 — AI cleaning explanations + executive summary
- [x] Phase 10 — AI business insights + feature engineering
- [x] Phase 11 — ML readiness + recommendation engine
- [x] Phase 12 — AI dataset chatbot
- [ ] Phase 13 — Report generation (PDF/HTML)
- [ ] Phase 14 — UI polish + Render deployment
- [ ] Phase 15 — Portfolio wrap-up
