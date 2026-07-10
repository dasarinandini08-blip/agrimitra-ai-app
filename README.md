# 🌾 AgriMitra — AI Agent for Smart Farming Advice

A RAG-ready, multi-language Streamlit application that gives farmers grounded,
conversational advice on weather, market (mandi) prices, and crop health.

This is a **demo/scaffold** build: all data services (weather, market prices,
knowledge retrieval) are simulated with realistic mock logic so the app runs
instantly with zero API keys. Clear placeholder functions are provided so you
can plug in real APIs and an LLM in minutes.

---

## ✨ Features

- **Multi-language UI**: English, Hindi, Spanish, Swahili (easy to extend)
- **Farmer profile sidebar**: Location, Soil Type, Current Crop
- **Live dashboard tabs**: Weather forecast, Mandi/market prices, Knowledge base
- **AI chat interface**: `st.chat_input` / `st.chat_message`, with session-persisted history
- **RAG-ready architecture**: simulated vector retrieval + LLM orchestration function,
  structured for easy integration with OpenAI, Google Gemini, or LangChain
- **Trusted sources footer**: builds farmer confidence in the data

---

## 📁 Project Structure

```
agrimitra_project/
├── app.py                        # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for environment-variable secrets
├── .gitignore                     # Keeps secrets & venvs out of git
├── .streamlit/
│   ├── config.toml                # App theme (greens/earth tones)
│   └── secrets.toml.example       # Template for Streamlit-native secrets
└── README.md                      # This file
```

---

## 🚀 Quick Start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API keys are required to explore
the demo — everything works out of the box with simulated data.

---

## 🔌 Plugging In Real APIs & an LLM

### 1. Set up your secrets (choose ONE method)

**Option A — Environment variables**
```bash
cp .env.example .env
# then edit .env and fill in your real keys
```
Load them in code with `os.environ.get("OPENAI_API_KEY")` (already wired up
in `app.py`) or with `python-dotenv`'s `load_dotenv()` if running locally.

**Option B — Streamlit secrets (recommended for Streamlit Cloud deployment)**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and fill in your real keys
```
Access them in code via `st.secrets["OPENAI_API_KEY"]`.

> ⚠️ **Never commit real keys.** `.env` and `.streamlit/secrets.toml` are
> already in `.gitignore`. Only the `.example` templates should be committed.

### 2. Swap the mock functions in `app.py` for real calls

| Function                     | Replace with                                              |
|-------------------------------|------------------------------------------------------------|
| `fetch_weather()`             | OpenWeatherMap / IMD API call                              |
| `fetch_market_prices()`       | Agmarknet / eNAM / data.gov.in Mandi price API              |
| `retrieve_context()`          | Real vector store (FAISS/Chroma/Pinecone) + embeddings      |
| `generate_farming_advice()`   | OpenAI, Google Gemini, or LangChain `RetrievalQA` chain      |

Each function already has commented, ready-to-uncomment example code showing
exactly how to make the swap — see the docstrings inside `app.py`.

---

## 🎨 Customization

- **Add a language**: add a new key to the `TRANSLATIONS` dict in `app.py`
  with the same fields as the existing languages.
- **Add knowledge base articles**: append entries to the `KNOWLEDGE_BASE` list
  (or point `retrieve_context()` at a real vector database).
- **Change theme colors**: edit `.streamlit/config.toml` and the `CUSTOM_CSS`
  block at the top of `app.py`.

---

## 🔒 Security Notes

- API keys are **never** hard-coded in `app.py` — they're loaded from
  environment variables or `st.secrets`.
- If a key is ever pasted into a chat, commit, or screenshot, treat it as
  compromised and rotate it immediately in your provider's dashboard.

---

## 📜 License

Use freely for personal, educational, or commercial projects.
