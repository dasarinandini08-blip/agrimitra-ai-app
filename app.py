"""
===============================================================================
 AI AGENT FOR SMART FARMING ADVICE  |  RAG-Ready Streamlit Application
===============================================================================
Run with:  streamlit run app.py

WHAT THIS FILE CONTAINS
------------------------------------------------------------------------------
1. Configuration & constants (languages, theme colors, mock knowledge base)
2. Simulated backend services:
     - Vector DB / RAG retriever      -> retrieve_context()
     - Weather API                    -> fetch_weather()
     - Mandi / Market Price API       -> fetch_market_prices()
     - LLM + RAG orchestration        -> generate_farming_advice()
3. Streamlit UI:
     - Sidebar farmer profile (Location, Soil Type, Crop)
     - Top language switcher
     - Dashboard tabs (Weather, Market Prices, Knowledge Base)
     - Chat interface (st.chat_input / st.chat_message)
     - Trusted sources footer

HOW TO PLUG IN REAL APIs LATER
------------------------------------------------------------------------------
- Replace the body of `fetch_weather()` with a real call (e.g. OpenWeatherMap,
  IMD API). Keep the same return dict shape so the UI doesn't need changes.
- Replace `fetch_market_prices()` with a real Mandi/Agmarknet API call.
- Replace `retrieve_context()` with a real vector store lookup (e.g. FAISS,
  Chroma, Pinecone) using embeddings from OpenAI/Google/HuggingFace.
- Replace `generate_farming_advice()` internals with an actual LLM call
  (OpenAI, Google Gemini, Anthropic, or a LangChain RetrievalQA chain).
  NEVER hard-code API keys in this file. Use environment variables or
  st.secrets, as shown in the placeholder function below.
===============================================================================
"""

import os
import random
import difflib
import base64
from datetime import datetime

import streamlit as st

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

# ==============================================================================
# 1. PAGE CONFIG & THEME
# ==============================================================================
st.set_page_config(
    page_title="AgriMitra | AI Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS: Agricultural theme (greens, earthy tones, clean background) ---
CUSTOM_CSS = """
<style>
    /* Overall app background */
    .stApp {
        background-color: #F7F6F2; /* clean light gray/off-white */
    }

    /* Headings */
    h1, h2, h3 {
        color: #2E5E32; /* deep forest green */
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0DED8;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #EFF3EA; /* soft sage */
        border-right: 1px solid #DDE3D4;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4C7A3F;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3B5F30;
        color: white;
    }

    /* Trusted sources footer */
    .trusted-footer {
        background-color: #EFF3EA;
        border-top: 2px solid #B8CBA6;
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 30px;
        font-size: 0.85rem;
        color: #4A4A4A;
    }

    .badge {
        display: inline-block;
        background-color: #DCE9D5;
        color: #2E5E32;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 2. MULTI-LANGUAGE SUPPORT
# ==============================================================================
# All user-facing strings live in this dictionary so the UI can be translated
# by simply switching the `LANG` key. In production, swap this for a proper
# i18n library (e.g. gettext, or a translation API call) if you need more
# languages or dynamic machine translation.

TRANSLATIONS = {
    "English": {
        "app_title": "🌾 AgriMitra — AI Smart Farming Assistant",
        "app_subtitle": "Your trusted digital companion for weather, market prices, and crop advice.",
        "sidebar_header": "👨‍🌾 Farmer Profile",
        "location_label": "📍 Location (State/Region)",
        "soil_label": "🧱 Soil Type",
        "crop_label": "🌱 Current Crop",
        "apply_button": "Update My Farm Context",
        "tab_weather": "🌤️ Weather",
        "tab_market": "💰 Market Prices",
        "tab_knowledge": "📚 Farming Knowledge",
        "chat_header": "💬 Ask AgriMitra",
        "chat_placeholder": "Ask about weather, crop prices, pests, soil health...",
        "weather_header": "Today's Weather Forecast",
        "market_header": "Today's Mandi Prices",
        "knowledge_header": "Localized Agricultural Knowledge Base",
        "temp": "Temperature",
        "humidity": "Humidity",
        "rain": "Rain Probability",
        "footer_title": "✅ Trusted Sources",
        "welcome_msg": "Namaste! I'm AgriMitra 🌾. Ask me about weather, mandi prices, pests, or soil health for your farm.",
        "no_match_msg": "I don't have specific information on that yet. Here's what I can help with: today's weather, mandi (market) prices, pest & soil health tips, or crop calendars. You can also reach out to your local Krishi Vigyan Kendra (KVK) for on-site guidance.",
        "upload_label": "📷 Upload a photo of your crop, leaf, pest, or soil (optional)",
        "upload_hint": "Attach one or more photos, then type your question and send. AgriMitra will look at the photo and tell you what it sees.",
        "image_offline_msg": "📷 I can see you've attached a photo, but photo analysis needs the live AI connection, which isn't set up right now. Please ask your question in text for now, or check back once the ANTHROPIC_API_KEY is configured.",
    },
    "Hindi": {
        "app_title": "🌾 एग्रीमित्र — एआई स्मार्ट खेती सहायक",
        "app_subtitle": "मौसम, बाजार भाव और फसल सलाह के लिए आपका विश्वसनीय डिजिटल साथी।",
        "sidebar_header": "👨‍🌾 किसान प्रोफ़ाइल",
        "location_label": "📍 स्थान (राज्य/क्षेत्र)",
        "soil_label": "🧱 मिट्टी का प्रकार",
        "crop_label": "🌱 वर्तमान फसल",
        "apply_button": "मेरी खेत जानकारी अपडेट करें",
        "tab_weather": "🌤️ मौसम",
        "tab_market": "💰 बाजार भाव",
        "tab_knowledge": "📚 कृषि ज्ञान",
        "chat_header": "💬 एग्रीमित्र से पूछें",
        "chat_placeholder": "मौसम, फसल भाव, कीट, मिट्टी स्वास्थ्य के बारे में पूछें...",
        "weather_header": "आज का मौसम पूर्वानुमान",
        "market_header": "आज के मंडी भाव",
        "knowledge_header": "स्थानीय कृषि ज्ञान आधार",
        "temp": "तापमान",
        "humidity": "नमी",
        "rain": "बारिश की संभावना",
        "footer_title": "✅ विश्वसनीय स्रोत",
        "welcome_msg": "नमस्ते! मैं एग्रीमित्र हूँ 🌾। अपने खेत के लिए मौसम, मंडी भाव, कीट या मिट्टी के स्वास्थ्य के बारे में पूछें।",
        "no_match_msg": "मेरे पास अभी इसके बारे में विशेष जानकारी नहीं है। मैं इनमें मदद कर सकता हूँ: आज का मौसम, मंडी भाव, कीट व मिट्टी स्वास्थ्य सुझाव, या फसल कैलेंडर। अधिक जानकारी के लिए अपने निकटतम कृषि विज्ञान केंद्र (KVK) से संपर्क करें।",
        "upload_label": "📷 अपनी फसल, पत्ती, कीट, या मिट्टी की फोटो अपलोड करें (वैकल्पिक)",
        "upload_hint": "एक या अधिक फोटो जोड़ें, फिर अपना सवाल टाइप करके भेजें। एग्रीमित्र फोटो देखकर बताएगा कि उसमें क्या दिख रहा है।",
        "image_offline_msg": "📷 मुझे दिख रहा है कि आपने एक फोटो जोड़ी है, लेकिन फोटो विश्लेषण के लिए लाइव एआई कनेक्शन चाहिए, जो अभी सेट नहीं है। कृपया अभी अपना सवाल टेक्स्ट में पूछें, या ANTHROPIC_API_KEY सेट होने के बाद दोबारा कोशिश करें।",
    },
    "Telugu": {
        "app_title": "🌾 అగ్రిమిత్ర — AI స్మార్ట్ వ్యవసాయ సహాయకుడు",
        "app_subtitle": "వాతావరణం, మార్కెట్ ధరలు మరియు పంట సలహాల కోసం మీ నమ్మకమైన డిజిటల్ సహచరుడు.",
        "sidebar_header": "👨‍🌾 రైతు ప్రొఫైల్",
        "location_label": "📍 ప్రాంతం (రాష్ట్రం/ప్రాంతం)",
        "soil_label": "🧱 నేల రకం",
        "crop_label": "🌱 ప్రస్తుత పంట",
        "apply_button": "నా వ్యవసాయ వివరాలను నవీకరించండి",
        "tab_weather": "🌤️ వాతావరణం",
        "tab_market": "💰 మార్కెట్ ధరలు",
        "tab_knowledge": "📚 వ్యవసాయ పరిజ్ఞానం",
        "chat_header": "💬 అగ్రిమిత్రను అడగండి",
        "chat_placeholder": "వాతావరణం, పంట ధరలు, తెగుళ్లు, నేల ఆరోగ్యం గురించి అడగండి...",
        "weather_header": "నేటి వాతావరణ సూచన",
        "market_header": "నేటి మండి ధరలు",
        "knowledge_header": "స్థానిక వ్యవసాయ పరిజ్ఞాన ఆధారం",
        "temp": "ఉష్ణోగ్రత",
        "humidity": "తేమ",
        "rain": "వర్షం అవకాశం",
        "footer_title": "✅ నమ్మదగిన మూలాలు",
        "welcome_msg": "నమస్తే! నేను అగ్రిమిత్ర 🌾. మీ పొలం కోసం వాతావరణం, మండి ధరలు, తెగుళ్లు లేదా నేల ఆరోగ్యం గురించి నన్ను అడగండి.",
        "no_match_msg": "దీని గురించి నా వద్ద ప్రత్యేక సమాచారం ఇంకా లేదు. నేను వీటిలో సహాయపడగలను: నేటి వాతావరణం, మండి ధరలు, తెగుళ్లు & నేల ఆరోగ్య చిట్కాలు, లేదా పంట క్యాలెండర్. మరింత సమాచారం కోసం మీ సమీప కృషి విజ్ఞాన కేంద్రం (KVK)ను సంప్రదించండి.",
        "upload_label": "📷 మీ పంట, ఆకు, తెగులు లేదా నేల ఫోటోను అప్‌లోడ్ చేయండి (ఐచ్ఛికం)",
        "upload_hint": "ఒకటి లేదా అంతకంటే ఎక్కువ ఫోటోలను జోడించి, మీ ప్రశ్నను టైప్ చేసి పంపండి. అగ్రిమిత్ర ఫోటోను చూసి అందులో ఏముందో చెబుతుంది.",
        "image_offline_msg": "📷 మీరు ఒక ఫోటో జోడించారని నాకు కనిపిస్తోంది, కానీ ఫోటో విశ్లేషణకు లైవ్ AI కనెక్షన్ అవసరం, ఇది ప్రస్తుతం సెటప్ చేయలేదు. దయచేసి ప్రస్తుతానికి మీ ప్రశ్నను టెక్స్ట్‌లో అడగండి, లేదా ANTHROPIC_API_KEY సెట్ చేసిన తర్వాత మళ్లీ ప్రయత్నించండి.",
    },
    "Spanish": {
        "app_title": "🌾 AgriMitra — Asistente Agrícola con IA",
        "app_subtitle": "Tu compañero digital de confianza para el clima, precios de mercado y consejos de cultivo.",
        "sidebar_header": "👨‍🌾 Perfil del Agricultor",
        "location_label": "📍 Ubicación (Estado/Región)",
        "soil_label": "🧱 Tipo de Suelo",
        "crop_label": "🌱 Cultivo Actual",
        "apply_button": "Actualizar mi Contexto Agrícola",
        "tab_weather": "🌤️ Clima",
        "tab_market": "💰 Precios de Mercado",
        "tab_knowledge": "📚 Conocimiento Agrícola",
        "chat_header": "💬 Pregúntale a AgriMitra",
        "chat_placeholder": "Pregunta sobre clima, precios, plagas, salud del suelo...",
        "weather_header": "Pronóstico del Clima de Hoy",
        "market_header": "Precios de Mercado de Hoy",
        "knowledge_header": "Base de Conocimiento Agrícola Local",
        "temp": "Temperatura",
        "humidity": "Humedad",
        "rain": "Probabilidad de Lluvia",
        "footer_title": "✅ Fuentes Confiables",
        "welcome_msg": "¡Hola! Soy AgriMitra 🌾. Pregúntame sobre el clima, precios de mercado, plagas o salud del suelo para tu granja.",
        "no_match_msg": "Todavía no tengo información específica sobre eso. Puedo ayudarte con: el clima de hoy, precios de mercado, consejos sobre plagas y salud del suelo, o calendarios de cultivo. También puedes contactar a tu Krishi Vigyan Kendra (KVK) local para orientación en el sitio.",
        "upload_label": "📷 Sube una foto de tu cultivo, hoja, plaga o suelo (opcional)",
        "upload_hint": "Adjunta una o más fotos, luego escribe tu pregunta y envíala. AgriMitra observará la foto y te dirá qué ve.",
        "image_offline_msg": "📷 Veo que has adjuntado una foto, pero el análisis de fotos necesita la conexión de IA en vivo, que no está configurada ahora mismo. Por favor haz tu pregunta en texto por ahora, o vuelve cuando ANTHROPIC_API_KEY esté configurado.",
    },
    "Swahili": {
        "app_title": "🌾 AgriMitra — Msaidizi wa Kilimo wa AI",
        "app_subtitle": "Rafiki yako wa kidijitali wa kuaminika kwa hali ya hewa, bei za soko, na ushauri wa mazao.",
        "sidebar_header": "👨‍🌾 Wasifu wa Mkulima",
        "location_label": "📍 Mahali (Jimbo/Eneo)",
        "soil_label": "🧱 Aina ya Udongo",
        "crop_label": "🌱 Zao la Sasa",
        "apply_button": "Sasisha Muktadha wa Shamba Langu",
        "tab_weather": "🌤️ Hali ya Hewa",
        "tab_market": "💰 Bei za Soko",
        "tab_knowledge": "📚 Maarifa ya Kilimo",
        "chat_header": "💬 Muulize AgriMitra",
        "chat_placeholder": "Uliza kuhusu hali ya hewa, bei za mazao, wadudu, afya ya udongo...",
        "weather_header": "Utabiri wa Hali ya Hewa wa Leo",
        "market_header": "Bei za Soko za Leo",
        "knowledge_header": "Hazina ya Maarifa ya Kilimo cha Eneo",
        "temp": "Joto",
        "humidity": "Unyevu",
        "rain": "Uwezekano wa Mvua",
        "footer_title": "✅ Vyanzo Vinavyoaminika",
        "welcome_msg": "Habari! Mimi ni AgriMitra 🌾. Niulize kuhusu hali ya hewa, bei za soko, wadudu, au afya ya udongo kwa shamba lako.",
        "no_match_msg": "Sina taarifa maalum kuhusu hilo bado. Ninaweza kusaidia na: hali ya hewa ya leo, bei za soko, vidokezo vya wadudu na afya ya udongo, au kalenda za mazao. Unaweza pia kuwasiliana na Krishi Vigyan Kendra (KVK) ya karibu nawe kwa ushauri wa moja kwa moja.",
        "upload_label": "📷 Pakia picha ya zao, jani, wadudu, au udongo wako (hiari)",
        "upload_hint": "Ambatanisha picha moja au zaidi, kisha andika swali lako na utume. AgriMitra itaangalia picha na kukuambia inaona nini.",
        "image_offline_msg": "📷 Naona umeambatanisha picha, lakini uchambuzi wa picha unahitaji muunganisho wa AI wa moja kwa moja, ambao haujawekwa sasa hivi. Tafadhali uliza swali lako kwa maandishi kwa sasa, au angalia tena baada ya ANTHROPIC_API_KEY kuwekwa.",
    },
}

# ==============================================================================
# 3. SESSION STATE INITIALIZATION
# ==============================================================================
if "language" not in st.session_state:
    st.session_state.language = "English"

if "location" not in st.session_state:
    st.session_state.location = "Telangana"

if "soil_type" not in st.session_state:
    st.session_state.soil_type = "Loam"

if "crop" not in st.session_state:
    st.session_state.crop = "Tomato"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}


def t(key: str) -> str:
    """Shortcut translation lookup for the currently selected language."""
    return TRANSLATIONS[st.session_state.language].get(
        key, TRANSLATIONS["English"].get(key, key)
    )


# ==============================================================================
# 4. SIMULATED BACKEND SERVICES (Mocked for Demo — replace with real APIs)
# ==============================================================================

# ------------------------------------------------------------------------------
# 4a. Simulated Vector DB / RAG Knowledge Base
# ------------------------------------------------------------------------------
# In production this would be embeddings stored in FAISS / Chroma / Pinecone,
# queried via similarity search. Here we simulate it with a small dictionary of
# curated agricultural documents, tagged by topic, so the retrieval logic and
# UI can be fully demonstrated end-to-end.

KNOWLEDGE_BASE = [
    {
        "id": "kb001",
        "topic": "Soil Health",
        "title": "Improving Loamy Soil Fertility",
        "content": (
            "Loamy soil retains nutrients well but benefits from annual organic "
            "compost addition (2-3 tons/acre) and crop rotation with legumes to "
            "replenish nitrogen naturally."
        ),
    },
    {
        "id": "kb002",
        "topic": "Soil Health",
        "title": "Managing Clay Soil Drainage",
        "content": (
            "Clay soil retains water and can cause root rot. Add gypsum and organic "
            "matter, and create raised beds to improve drainage for sensitive crops."
        ),
    },
    {
        "id": "kb003",
        "topic": "Pest Management",
        "title": "Yellowing Corn Leaves — Common Causes",
        "content": (
            "Yellowing corn leaves are commonly caused by nitrogen deficiency, "
            "overwatering, or fall armyworm infestation. Check the underside of "
            "leaves for larvae; if none, apply a nitrogen-rich fertilizer (urea) "
            "in split doses."
        ),
    },
    {
        "id": "kb004",
        "topic": "Pest Management",
        "title": "Tomato Leaf Curl Prevention",
        "content": (
            "Tomato leaf curl virus spreads via whiteflies. Use yellow sticky traps, "
            "neem oil spray every 7-10 days, and remove infected plants immediately "
            "to prevent spread."
        ),
    },
    {
        "id": "kb005",
        "topic": "Crop Calendar",
        "title": "Tomato Planting Calendar",
        "content": (
            "In tropical and subtropical regions, tomatoes are best transplanted "
            "at the start of the post-monsoon season (Oct-Nov) or early spring "
            "(Feb-Mar) to avoid peak pest pressure and excess rainfall."
        ),
    },
    {
        "id": "kb006",
        "topic": "Crop Calendar",
        "title": "Potato Sowing Window",
        "content": (
            "Potatoes thrive when sown in cool weather, typically October to "
            "November in North India, with harvest 90-120 days later."
        ),
    },
    {
        "id": "kb007",
        "topic": "Irrigation",
        "title": "Drip Irrigation for Water Savings",
        "content": (
            "Drip irrigation can reduce water usage by 30-50% compared to flood "
            "irrigation, while also reducing weed growth and fungal disease risk."
        ),
    },
    {
        "id": "kb008",
        "topic": "Support Services",
        "title": "What is a Krishi Vigyan Kendra (KVK)?",
        "content": (
            "A Krishi Vigyan Kendra (KVK) is a district-level farm science center, "
            "under ICAR, that offers free on-site soil testing, crop demonstrations, "
            "training, and expert advice tailored to your local conditions. Farmers "
            "can walk in or call their district KVK office directly for personalized "
            "guidance beyond what a general assistant like AgriMitra can offer."
        ),
    },
]


def _fuzzy_contains(word: str, targets, cutoff: float = 0.8) -> bool:
    """
    Typo-tolerant check: returns True if `word` closely matches any string in
    `targets` (e.g. "wheather" ~ "weather"). Uses difflib's SequenceMatcher
    ratio under the hood via get_close_matches.
    """
    if len(word) < 4:
        return word in targets
    return bool(difflib.get_close_matches(word, targets, n=1, cutoff=cutoff))


def retrieve_context(query: str, crop: str, soil_type: str, top_k: int = 3):
    """
    SIMULATED RAG RETRIEVER
    ------------------------------------------------------------------------
    Mimics a vector similarity search over the knowledge base.
    Replace this with a real embedding-based retriever, e.g.:

        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings

        vectorstore = FAISS.load_local("agri_index", OpenAIEmbeddings())
        docs = vectorstore.similarity_search(query, k=top_k)
        return docs

    For this demo, we do simple keyword overlap scoring (with light typo
    tolerance) so the app is fully runnable offline without any API keys.

    Returns a tuple: (top_docs, best_score). `best_score` lets the caller
    know how confident the match is, so it can fall back gracefully instead
    of always returning the same crop/soil-based default.
    """
    query_lower = query.lower()
    query_words = query_lower.split()
    scored_docs = []

    for doc in KNOWLEDGE_BASE:
        score = 0
        haystack_text = (doc["title"] + " " + doc["content"] + " " + doc["topic"]).lower()
        haystack_words = set(haystack_text.replace("(", " ").replace(")", " ").split())

        for word in query_words:
            cleaned = word.strip(".,?!")
            if len(cleaned) > 3:
                if cleaned in haystack_text:
                    score += 1
                elif _fuzzy_contains(cleaned, haystack_words):
                    score += 1  # typo-tolerant partial credit

        # Boost score if it matches the farmer's current crop or soil type
        if crop.lower() in haystack_text:
            score += 2
        if soil_type.lower() in haystack_text:
            score += 2

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:top_k]]
    best_score = scored_docs[0][0] if scored_docs else 0

    return top_docs, best_score


# ------------------------------------------------------------------------------
# 4b. Simulated Weather API
# ------------------------------------------------------------------------------
def fetch_weather(location: str):
    """
    SIMULATED WEATHER API CALL
    ------------------------------------------------------------------------
    Replace with a real call, e.g.:

        import requests
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": API_KEY, "units": "metric"},
        )
        data = resp.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain_probability": data.get("rain", {}).get("1h", 0),
            "condition": data["weather"][0]["description"],
        }

    Here we simulate realistic-looking values seeded by location name so
    results stay consistent within a session.
    """
    random.seed(hash(location) % (10**6))
    temperature = round(random.uniform(22, 38), 1)
    humidity = random.randint(40, 90)
    rain_probability = random.randint(0, 100)
    conditions = ["Clear Sky", "Partly Cloudy", "Overcast", "Light Rain", "Humid & Hot"]
    condition = random.choice(conditions)

    return {
        "temperature": temperature,
        "humidity": humidity,
        "rain_probability": rain_probability,
        "condition": condition,
        "updated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


# ------------------------------------------------------------------------------
# 4c. Simulated Mandi / Market Price API
# ------------------------------------------------------------------------------
def fetch_market_prices(location: str):
    """
    SIMULATED MANDI/MARKET PRICE API CALL
    ------------------------------------------------------------------------
    Replace with a real call to a government Agmarknet / eNAM API, e.g.:

        import requests
        resp = requests.get(
            "https://api.data.gov.in/resource/mandi-prices",
            params={"api-key": os.environ.get("MANDI_API_KEY"), "state": location},
        )
        return resp.json()["records"]

    Here we simulate plausible commodity prices (INR per quintal) seeded by
    location so values stay stable within a session but vary by region.
    """
    random.seed((hash(location) + 7) % (10**6))
    commodities = {
        "Tomato": random.randint(800, 2500),
        "Potato": random.randint(600, 1800),
        "Onion": random.randint(900, 2200),
        "Wheat": random.randint(2000, 2600),
        "Rice (Paddy)": random.randint(1800, 2400),
    }
    # Simulate day-over-day change percentage for trend arrows
    trends = {k: round(random.uniform(-8, 8), 1) for k in commodities}

    return {
        "prices": commodities,
        "trends": trends,
        "unit": "₹ per Quintal",
        "updated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


# ------------------------------------------------------------------------------
# 4d. LLM + RAG Orchestration (Placeholder — plug in your API key here)
# ------------------------------------------------------------------------------

# Keyword sets used for lightweight intent detection. Each list includes
# common variants/typos are handled via fuzzy matching in `_matches_intent`.
WEATHER_KEYWORDS = ["weather", "rain", "forecast", "temperature", "climate", "humidity", "wind", "sunny", "monsoon"]
PRICE_KEYWORDS = ["price", "prices", "rate", "rates", "market", "mandi", "cost", "sell", "selling"]
KVK_KEYWORDS = ["kvk", "krishi", "vigyan", "kendra", "extension", "officer", "helpline", "contact"]


def _matches_intent(query_lower: str, keywords, cutoff: float = 0.82) -> bool:
    """
    Typo-tolerant intent check: True if any word in the query closely matches
    any keyword in `keywords` (exact substring OR fuzzy match, so "wheather"
    still triggers the weather intent even though it's misspelled).
    """
    words = [w.strip(".,?!") for w in query_lower.split()]
    for kw in keywords:
        if kw in query_lower:
            return True
        for w in words:
            if len(w) >= 4 and difflib.SequenceMatcher(None, w, kw).ratio() >= cutoff:
                return True
    return False


def generate_farming_advice(query: str, context_docs: list, best_score: int, location: str,
                             soil_type: str, crop: str, weather: dict,
                             market: dict, language: str, images: list = None) -> str:
    """
    LLM + RAG ORCHESTRATION FUNCTION
    ------------------------------------------------------------------------
    This is the core "AI Agent" brain. It assembles retrieved context
    (from retrieve_context) plus live farm data (weather, market prices)
    into a prompt, then calls an LLM to generate grounded advice.

    ---- LIVE LLM: Anthropic Claude (claude-sonnet-4-6) ----
    When an ANTHROPIC_API_KEY is set (see setup notes at the bottom of this
    file), this function calls the real Claude API with the farmer's question
    plus full grounding context (weather, market prices, soil/crop profile,
    and the retrieved knowledge-base snippets) so it can answer genuinely
    open-ended farming questions — not just the handful of pre-built intents.

    IMPORTANT: Always load API keys from environment variables or
    st.secrets["ANTHROPIC_API_KEY"] — never paste raw key strings into source code.

    ---- OFFLINE FALLBACK ----
    If no API key is configured, the `anthropic` package isn't installed, or
    the live call fails for any reason, this function falls back to a
    rule-based, templated response built from the retrieved knowledge base
    documents and live mock data, so the app still works without any keys.
    Intent detection there is typo-tolerant (see _matches_intent) and there
    is an explicit "no confident match" message so the offline mode doesn't
    silently repeat the same crop/soil answer for unrelated questions.

    `images` (optional): list of dicts {"media_type": "image/jpeg", "data": <base64 str>}.
    When present and a live API key is configured, Claude's vision capability
    is used to actually look at the photo(s) — e.g. diagnosing a leaf disease
    or pest from an uploaded picture. Without a live key, image analysis
    isn't available and the farmer is told to ask in text instead.
    """

    # Build the "system prompt" that would be sent to a real LLM.
    # Even in demo mode we construct it, so you can print/log it while testing.
    context_text = "\n".join([f"- {d['title']}: {d['content']}" for d in context_docs])

    system_prompt = f"""
    You are AgriMitra, a friendly and knowledgeable AI farming advisor.
    Respond in {language}. Be concise, practical, and encouraging.

    Farmer Context:
    - Location: {location}
    - Soil Type: {soil_type}
    - Current Crop: {crop}
    - Weather: {weather['temperature']}°C, {weather['humidity']}% humidity,
      {weather['rain_probability']}% rain chance, {weather['condition']}
    - Market Prices ({market['unit']}): {market['prices']}

    Retrieved Knowledge Base Context:
    {context_text}

    Farmer's Question: {query}
    """

    # ---- Real LLM call: Anthropic Claude ----
    # Reads the key from an environment variable first, then falls back to
    # Streamlit secrets (st.secrets["ANTHROPIC_API_KEY"] in .streamlit/secrets.toml).
    # NEVER hard-code the key in source — see setup notes at the bottom of this file.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""

    if api_key and ANTHROPIC_SDK_AVAILABLE:
        try:
            # Build multimodal content: any attached photos first, then the text
            # prompt. Claude's vision capability lets it actually look at crop,
            # leaf, pest, or soil photos and describe what it sees / diagnose issues.
            content_blocks = []
            for img in (images or []):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                })
            image_instruction = (
                "\n\nThe farmer has also attached one or more photos above. Look at "
                "them carefully and describe what you observe (crop/leaf condition, "
                "visible pests, soil appearance, etc.), then give a diagnosis or "
                "practical next steps."
                if images else ""
            )
            content_blocks.append({"type": "text", "text": system_prompt + image_instruction})

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=(
                    "You are AgriMitra, a friendly, practical AI farming advisor for "
                    "smallholder farmers. Always answer grounded in the farmer context, "
                    "live weather, live market prices, and knowledge base notes given to "
                    "you below. When the farmer attaches photos, examine them closely "
                    "and describe specifically what you see before giving advice — name "
                    "visible symptoms, pests, or soil conditions rather than speaking "
                    "generically. Be concise (3-6 short paragraphs or a short bullet "
                    "list), practical, and encouraging. If the farmer's question falls "
                    "outside farming, weather, market, or rural-livelihood topics, "
                    "gently redirect them back to what you can help with. Always "
                    f"respond in {language}."
                ),
                messages=[{"role": "user", "content": content_blocks}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            )
        except Exception as e:
            # Fall through to the rule-based demo behavior below, but let the
            # farmer know the AI call didn't go through rather than failing silently.
            st.warning(f"⚠️ Live AI call failed ({e}); showing offline demo answer instead.")

    # ---- Demo fallback (used only if no API key / SDK / call failure): rule-based templated advice ----
    # The rule-based fallback below has no vision capability, so if the farmer
    # attached photos, be upfront about that instead of silently ignoring them.
    if images:
        return t("image_offline_msg")

    query_lower = query.lower()

    # 1. Market / price intent (typo-tolerant)
    if _matches_intent(query_lower, PRICE_KEYWORDS):
        lines = [f"💰 Here are today's mandi prices for **{location}** ({market['unit']}):"]
        for commodity, price in market["prices"].items():
            trend = market["trends"][commodity]
            arrow = "📈" if trend >= 0 else "📉"
            lines.append(f"- {commodity}: ₹{price} {arrow} ({trend:+.1f}% vs yesterday)")
        return "\n".join(lines)

    # 2. Weather intent (typo-tolerant, so "wheather" still matches)
    if _matches_intent(query_lower, WEATHER_KEYWORDS):
        return (
            f"🌤️ Today's forecast for **{location}**: {weather['condition']}, "
            f"🌡️ {weather['temperature']}°C, 💧 {weather['humidity']}% humidity, "
            f"🌧️ {weather['rain_probability']}% chance of rain.\n\n"
            f"{'💡 Consider delaying irrigation today due to expected rainfall.' if weather['rain_probability'] > 60 else '💡 Rain is unlikely — plan irrigation as needed.'}"
        )

    # 3. KVK / support-service intent
    if _matches_intent(query_lower, KVK_KEYWORDS):
        kvk_doc = next((d for d in KNOWLEDGE_BASE if d["id"] == "kb008"), None)
        if kvk_doc:
            return f"🏛️ **{kvk_doc['title']}**\n\n{kvk_doc['content']}"

    # 4. Confident knowledge-base match → combine retrieved docs into advice
    if best_score >= 2 and context_docs:
        advice_lines = [f"🌱 Based on your **{crop}** crop on **{soil_type}** soil in **{location}**, here's what I found:\n"]
        for doc in context_docs:
            advice_lines.append(f"**{doc['title']}** ({doc['topic']})\n{doc['content']}\n")
        advice_lines.append(
            "👉 If symptoms persist, consult your local Krishi Vigyan Kendra (KVK) "
            "or agricultural extension officer for an on-site inspection."
        )
        return "\n".join(advice_lines)

    # 5. No confident match anywhere → say so explicitly instead of repeating
    #    the same generic crop/soil advice for every unrelated question.
    return t("no_match_msg")


# ==============================================================================
# 5. UI — TOP BAR (Title + Language Selector)
# ==============================================================================
top_left, top_right = st.columns([4, 1])

with top_right:
    selected_lang = st.selectbox(
        "🌐 Language / भाषा / భాష",
        options=list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language),
    )
    st.session_state.language = selected_lang

with top_left:
    st.title(t("app_title"))
    st.caption(t("app_subtitle"))

st.divider()

# ==============================================================================
# 6. SIDEBAR — Farmer Profile / Context Widgets
# ==============================================================================
with st.sidebar:
    st.header(t("sidebar_header"))

    st.session_state.location = st.selectbox(
        t("location_label"),
        options=["Telangana", "Punjab", "Maharashtra", "Uttar Pradesh", "Karnataka", "Bihar", "Tamil Nadu"],
        index=["Telangana", "Punjab", "Maharashtra", "Uttar Pradesh", "Karnataka", "Bihar", "Tamil Nadu"].index(
            st.session_state.location
        ),
    )

    st.session_state.soil_type = st.radio(
        t("soil_label"),
        options=["Clay", "Sandy", "Loam"],
        index=["Clay", "Sandy", "Loam"].index(st.session_state.soil_type),
        horizontal=True,
    )

    st.session_state.crop = st.selectbox(
        t("crop_label"),
        options=["Tomato", "Potato", "Onion", "Wheat", "Rice (Paddy)", "Corn", "Cotton"],
        index=["Tomato", "Potato", "Onion", "Wheat", "Rice (Paddy)", "Corn", "Cotton"].index(
            st.session_state.crop
        ) if st.session_state.crop in ["Tomato", "Potato", "Onion", "Wheat", "Rice (Paddy)", "Corn", "Cotton"] else 0,
    )

    st.button(t("apply_button"), use_container_width=True)

    st.markdown("---")
    st.caption(f"🧑‍🌾 {st.session_state.crop} | {st.session_state.soil_type} soil | 📍 {st.session_state.location}")

    st.markdown("---")
    with st.expander("🔧 AI Connection Status (debug)"):
        _debug_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not _debug_key:
            try:
                _debug_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                _debug_key = ""
        if _debug_key:
            st.success(f"✅ API key detected (starts with `{_debug_key[:12]}...`, length {len(_debug_key)})")
        else:
            st.error("❌ No API key detected in environment or st.secrets")
        if ANTHROPIC_SDK_AVAILABLE:
            st.success("✅ `anthropic` package is installed")
        else:
            st.error("❌ `anthropic` package is NOT installed — run: pip install anthropic")

# Fetch (simulated) live data based on current sidebar context.
weather_data = fetch_weather(st.session_state.location)
market_data = fetch_market_prices(st.session_state.location)

# ==============================================================================
# 7. DASHBOARD TABS — Weather | Market Prices | Knowledge Base
# ==============================================================================
tab_weather, tab_market, tab_knowledge = st.tabs(
    [t("tab_weather"), t("tab_market"), t("tab_knowledge")]
)

# ---- Weather Tab ----
with tab_weather:
    st.subheader(t("weather_header"))
    st.caption(f"📍 {st.session_state.location} · Updated {weather_data['updated_at']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"🌡️ {t('temp')}", f"{weather_data['temperature']} °C")
    col2.metric(f"💧 {t('humidity')}", f"{weather_data['humidity']}%")
    col3.metric(f"🌧️ {t('rain')}", f"{weather_data['rain_probability']}%")
    col4.metric("☁️ Condition", weather_data["condition"])

    if weather_data["rain_probability"] > 60:
        st.info("🌧️ High rain probability today — consider postponing irrigation and pesticide spraying.")
    elif weather_data["temperature"] > 35:
        st.warning("🔥 High temperature alert — ensure adequate irrigation to prevent heat stress on crops.")
    else:
        st.success("✅ Weather conditions look favorable for regular farm activities today.")

# ---- Market Prices Tab ----
with tab_market:
    st.subheader(t("market_header"))
    st.caption(f"📍 {st.session_state.location} · Updated {market_data['updated_at']} · Unit: {market_data['unit']}")

    price_cols = st.columns(len(market_data["prices"]))
    for col, (commodity, price) in zip(price_cols, market_data["prices"].items()):
        trend = market_data["trends"][commodity]
        col.metric(
            label=f"💰 {commodity}",
            value=f"₹{price}",
            delta=f"{trend:+.1f}%",
        )

# ---- Knowledge Base Tab ----
with tab_knowledge:
    st.subheader(t("knowledge_header"))
    st.caption("📚 Simulated retrieval-augmented knowledge base (soil, pests, crop calendars)")

    topics = sorted(set(doc["topic"] for doc in KNOWLEDGE_BASE))
    selected_topic = st.selectbox("Filter by topic:", options=["All"] + topics)

    for doc in KNOWLEDGE_BASE:
        if selected_topic != "All" and doc["topic"] != selected_topic:
            continue
        with st.container(border=True):
            st.markdown(f"**{doc['title']}**  \n🏷️ *{doc['topic']}*")
            st.write(doc["content"])

st.divider()

# ==============================================================================
# 8. CHAT INTERFACE — AI Agent Conversation
# ==============================================================================
st.subheader(t("chat_header"))

# Show a welcome message once, at the start of the session.
if not st.session_state.chat_history:
    st.session_state.chat_history.append({"role": "assistant", "content": t("welcome_msg")})

# Render existing conversation history (including any attached photos).
for message in st.session_state.chat_history:
    avatar = "🧑‍🌾" if message["role"] == "user" else "🌾"
    with st.chat_message(message["role"], avatar=avatar):
        for img in message.get("images", []):
            st.image(base64.b64decode(img["data"]), width=220)
        st.markdown(message["content"])

# Photo uploader — lets a farmer attach crop/leaf/pest/soil photos before asking.
# The key includes a counter so the widget resets (clears the selection) after
# each message is sent, instead of re-sending the same photo every turn.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    t("upload_label"),
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=f"crop_photo_uploader_{st.session_state.uploader_key}",
    help=t("upload_hint"),
)

# Chat input box.
user_query = st.chat_input(t("chat_placeholder"))

if user_query:
    # 1. Convert any attached photos to base64 for both display and the API call.
    image_payloads = []
    for f in (uploaded_files or []):
        raw_bytes = f.getvalue()
        media_type = f.type if f.type in ("image/png", "image/jpeg", "image/webp") else "image/jpeg"
        image_payloads.append({
            "media_type": media_type,
            "data": base64.b64encode(raw_bytes).decode("utf-8"),
        })

    # 2. Save & display the user's message (with photo thumbnails, if any).
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "images": image_payloads,
    })
    with st.chat_message("user", avatar="🧑‍🌾"):
        for img in image_payloads:
            st.image(base64.b64decode(img["data"]), width=220)
        st.markdown(user_query)

    # 3. Retrieve relevant context from the (simulated) vector knowledge base.
    retrieved_docs, best_score = retrieve_context(
        query=user_query,
        crop=st.session_state.crop,
        soil_type=st.session_state.soil_type,
    )

    # 4. Generate advice via the LLM + RAG function (with vision if photos attached).
    with st.chat_message("assistant", avatar="🌾"):
        with st.spinner("🌱 Thinking..."):
            answer = generate_farming_advice(
                query=user_query,
                context_docs=retrieved_docs,
                best_score=best_score,
                location=st.session_state.location,
                soil_type=st.session_state.soil_type,
                crop=st.session_state.crop,
                weather=weather_data,
                market=market_data,
                language=st.session_state.language,
                images=image_payloads,
            )
        st.markdown(answer)

    # 5. Save the assistant's reply to history.
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # 6. Reset the uploader so photos don't get re-attached to the next message.
    st.session_state.uploader_key += 1
    st.rerun()

# ==============================================================================
# 9. TRUSTED SOURCES FOOTER
# ==============================================================================
st.markdown(
    f"""
    <div class="trusted-footer">
        <strong>{t('footer_title')}</strong><br>
        <span class="badge">🌦️ India Meteorological Department (IMD)</span>
        <span class="badge">🏛️ Ministry of Agriculture &amp; Farmers Welfare</span>
        <span class="badge">📊 Agmarknet / eNAM Mandi Prices</span>
        <span class="badge">🌱 Krishi Vigyan Kendra (KVK)</span>
        <span class="badge">🔬 ICAR Research Advisories</span>
        <br><br>
        <em>Note: This demo uses simulated data for illustration. Connect live
        APIs (see code comments in app.py) for production deployment.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
