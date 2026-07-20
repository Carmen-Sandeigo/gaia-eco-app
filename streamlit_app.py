import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# --- 1. CONFIG & VISUAL FIXES ---
st.set_page_config(page_title="Gaia", page_icon="🌎", layout="wide")

# NOTE: the @ was missing before "wght" in the original — that's why the
# custom font wasn't loading and everything fell back to the browser default.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');

    *:not([data-testid="stIconMaterial"]):not(.material-icons) {
        font-family: 'Quicksand', sans-serif !important;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F4FFF5 !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, [data-testid="stMarkdownContainer"] p {
        color: #00241B !important;
    }

    .stTextArea textarea, .stTextArea textarea:disabled {
        color: #00241B !important;
        -webkit-text-fill-color: #00241B !important;
        background-color: #FFFFFF !important;
    }

    div.stButton > button, div.stButton > button p {
        background-color: #04724D !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    div.stButton > button:hover {
        background-color: #03583C !important;
        color: #FFFFFF !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #EAF7EA !important;
        border-radius: 12px !important;
        border: none !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessageContent"], .stChatMessage p {
        color: #00241B !important;
    }

    .stTextInput input, [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #00241B !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 12px !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #04724D !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #04724D !important;
        color: #00241B !important;
    }

    [data-testid="stMetric"], [data-testid="stMetricContainer"] {
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. DATA FILE LOADER (cached so it only runs once, not on every click) ---
@st.cache_data(show_spinner=False)
def load_text_files():
    files = {
        "knowledge_base": "envirobot_knowledge_base.txt",
        "fun_facts_base": "envirobot_fun_facts.txt",
        "instructions_text": "instructions.txt",
        "crafting_instructions_text": "crafting_instructions.txt",
        "disposal_instructions_text": "disposal_instructions.txt",
        "upscaling_instructions_text": "upscaling_instructions.txt",
    }
    contents = {}
    missing = []
    for key, filename in files.items():
        try:
            with open(filename, "r", encoding="utf-8") as file:
                contents[key] = file.read()
        except FileNotFoundError:
            missing.append(filename)
            contents[key] = ""
    return contents, missing


text_files, missing_files = load_text_files()
if missing_files:
    st.error(
        "These files are missing from the app folder, so responses will be "
        f"limited: {', '.join(missing_files)}. Make sure they're uploaded "
        "alongside app.py in your repo."
    )

knowledge_base = text_files["knowledge_base"]
fun_facts_base = text_files["fun_facts_base"]
instructions_text = text_files["instructions_text"]
crafting_instructions_text = text_files["crafting_instructions_text"]
disposal_instructions_text = text_files["disposal_instructions_text"]
upscaling_instructions_text = text_files["upscaling_instructions_text"]


# --- 3. RAG EMBEDDINGS BACKEND (cached so the model loads once) ---
def preprocess_text(text_list):
    cleaned_chunks = []
    for text in text_list:
        cleaned_text = text.strip()
        chunks = cleaned_text.split("\n")
        for chunk in chunks:
            new_chunk = chunk.strip()
            if new_chunk:
                cleaned_chunks.append(new_chunk)
    return cleaned_chunks


@st.cache_resource(show_spinner="Loading Gaia's brain...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def build_embeddings(_model, text_chunks):
    # _model is prefixed with underscore so Streamlit doesn't try to hash it
    return _model.encode(text_chunks, convert_to_tensor=True)


model = load_model()
cleaned_chunks = preprocess_text([knowledge_base, fun_facts_base])

if cleaned_chunks:
    chunk_embeddings = build_embeddings(model, cleaned_chunks)
else:
    chunk_embeddings = None


def get_top_chunks(query, chunk_embeddings, text_chunks, k=3):
    if chunk_embeddings is None or not text_chunks:
        return []
    # Guard against asking for more chunks than actually exist —
    # this was the likely source of your ValueError.
    k = min(k, len(text_chunks))
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
    top_indices = torch.topk(similarities, k=k).indices
    return [text_chunks[i] for i in top_indices]


# --- 4. HUGGING FACE INFERENCE CLIENT ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.warning(
        "No HF_TOKEN found in Streamlit secrets. Add it under "
        "Settings → Secrets as: HF_TOKEN = \"your_token_here\""
    )

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

DEFAULT_SYSTEM_PROMPT = (
    "You're an environmental chatbot that answers the user's questions. You ask the "
    "user what materials the user has and then give suggestions on what they can make "
    "using those materials to reuse it. If the user asks to dispose of the materials, "
    f"give suggestions on how to get rid of the materials in ways that are environmentally "
    f"sustainable. Use the following information for a response: {instructions_text}"
)
DISPOSAL_PROMPT = (
    "You're an environmental chatbot focused ONLY on disposal. Ask the user what "
    "materials they need to get rid of, then give specific, environmentally sustainable "
    f"disposal methods. This file has more details: {disposal_instructions_text}"
)
CRAFTING_PROMPT = (
    "You're an environmental chatbot focused ONLY on crafting/reuse. Ask the user what "
    "materials they have on hand, then suggest specific, creative DIY projects they can "
    f"make with those materials. This file has more details: {crafting_instructions_text}"
)
UPSCALING_PROMPT = (
    "You're an environmental chatbot focused ONLY on upcycling. Ask the user what "
    "materials they have, then suggest ways to transform those materials into something "
    f"more valuable or higher-use than their original form. This file has more details: {upscaling_instructions_text}"
)


def respond(message, history):
    top_results = get_top_chunks(message, chunk_embeddings, cleaned_chunks)

    if message == "Disposal":
        system_content = DISPOSAL_PROMPT
    elif message == "Crafting":
        system_content = CRAFTING_PROMPT
    elif message == "Upscaling":
        system_content = UPSCALING_PROMPT
    else:
        system_content = DEFAULT_SYSTEM_PROMPT

    ai_messages = [{
        "role": "system",
        "content": (
            f"{system_content} Give all responses in English. Do not use Chinese or "
            f"any other language besides English. Use the following information for a "
            f"response: {top_results}, {instructions_text}"
        ),
    }]

    for turn in history:
        ai_messages.append({"role": turn["role"], "content": turn["content"]})
    ai_messages.append({"role": "user", "content": message})

    try:
        response = client.chat_completion(ai_messages, max_tokens=350, temperature=1.0)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Gaia couldn't reach the model right now ({e}). Please try again."


# --- 5. INTERFACE ---
st.image("logo_banner.png", use_container_width=True)
st.title("🌎 Welcome to Gaia!")
st.write("#### Decide whether you'd like to dispose, reuse, or upcycle your items!")

if "eco_score" not in st.session_state:
    st.session_state.eco_score = 0
if "eco_logs" not in st.session_state:
    st.session_state.eco_logs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["💬 Chat with Gaia", "📊 Live Eco-Tracker"])

# ==========================================
# TAB 1: CHAT
# ==========================================
with tab1:
    st.caption("Conversations for a cleaner Earth!")

    def run_preset(preset_message):
        st.session_state.chat_history.append({"role": "user", "content": preset_message})
        clean_hist = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]
        with st.spinner("Gaia is typing..."):
            reply = respond(preset_message, clean_hist)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("📌 Disposal Setup", use_container_width=True, key="disposal_btn"):
        run_preset("Disposal")
        st.rerun()

    if col_ex2.button("✂️ Crafting Setup", use_container_width=True, key="crafting_btn"):
        run_preset("Crafting")
        st.rerun()

    if col_ex3.button("🏷️ Upscaling Setup", use_container_width=True, key="upscaling_btn"):
        run_preset("Upscaling")
        st.rerun()

    for chat in st.session_state.chat_history:
        avatar = "🧑" if chat["role"] == "user" else "🌎"
        with st.chat_message(chat["role"], avatar=avatar):
            st.write(chat["content"])

    if user_prompt := st.chat_input("Ask Gaia anything..."):
        with st.chat_message("user", avatar="🧑"):
            st.write(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant", avatar="🌎"):
            with st.spinner("Gaia is typing..."):
                clean_hist = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                ]
                bot_reply = respond(user_prompt, clean_hist)
                st.write(bot_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# ==========================================
# TAB 2: LIVE ECO-TRACKER
# ==========================================
with tab2:
    st.write("### Claim your Eco-Points here as you complete Gaia's recommendations!")

    score = st.session_state.eco_score
    if score >= 350:
        badge, progress = "🌍 Level 4: Eco Hero", 1.0
    elif score >= 250:
        badge, progress = "🌳 Level 3: Earth Guardian", (score - 250) / 100
    elif score >= 100:
        badge, progress = "🌿 Level 2: Green Sprout", (score - 100) / 150
    else:
        badge, progress = "🌱 Level 1: Eco Seedling", score / 100
    progress = min(max(progress, 0.0), 1.0)

    col_stats, col_btns = st.columns(2)

    with col_stats:
        st.metric(label="Your Total Eco-Points", value=f"{score} pts")
        st.progress(progress, text="Progress to Next Earth Badge")
        st.info(f"Your Current Status: **{badge}**")

    with col_btns:
        st.write("#### What did you complete today?")

        if st.button("✂️ Completed a DIY Craft (+25 pts)", use_container_width=True, key="craft_btn"):
            st.session_state.eco_score += 25
            st.session_state.eco_logs.append("✅ Completed A Diy Craft")
            st.rerun()

        if st.button("🗑️ Disposed of Materials Safely (+15 pts)", use_container_width=True, key="dispose_btn"):
            st.session_state.eco_score += 15
            st.session_state.eco_logs.append("✅ Disposed Of Materials Safely")
            st.rerun()

        if st.button("🏷️ Listed/Sold an Item (+20 pts)", use_container_width=True, key="sell_btn"):
            st.session_state.eco_score += 20
            st.session_state.eco_logs.append("✅ Listed/Sold An Item")
            st.rerun()

        if st.button("🎁 Donated Used Clothing (+20 pts)", use_container_width=True, key="donate_btn"):
            st.session_state.eco_score += 20
            st.session_state.eco_logs.append("✅ Donated Used Clothing")
            st.rerun()

    st.write("### 📜 Your Green Activity Log")
    log_history_text = (
        "\n".join(st.session_state.eco_logs[::-1])
        if st.session_state.eco_logs
        else "No activities logged yet. Start completing tasks!"
    )
    st.text_area("History", value=log_history_text, height=150, disabled=True)
