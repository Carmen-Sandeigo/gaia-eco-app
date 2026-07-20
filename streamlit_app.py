import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from datetime import datetime

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


# Candidate (model, provider) pairs to try in order for the "identify a material
# from a photo" feature. Providers can be flaky or lack a given model, so we
# fall through the list instead of hard-coding just one.
VISION_CANDIDATES = [
    ("CohereLabs/command-a-vision-07-2025", "cohere"),
    ("zai-org/GLM-4.5V", None),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", None),
]


def identify_material_from_image(image_bytes, mime_type="image/jpeg"):
    """Send a photo to a vision model and get back a short material description.
    Tries several models/providers in order. Returns (description, error_message)
    — exactly one of them will be set."""
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Look at this photo of an item someone wants to reuse, "
                    "recycle, or dispose of. In one short sentence, name the "
                    "item and the main material(s) it's made of (e.g. "
                    "'a glass jar' or 'a cardboard box with plastic tape'). "
                    "Don't add any extra commentary."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ],
    }]

    errors = []
    for model_id, provider in VISION_CANDIDATES:
        try:
            candidate_client = InferenceClient(
                model_id, token=HF_TOKEN, provider=provider
            ) if provider else InferenceClient(model_id, token=HF_TOKEN)

            response = candidate_client.chat_completion(
                messages=messages, max_tokens=300, temperature=0.4
            )
            message = response.choices[0].message
            content = (message.content or "").strip()
            if not content:
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    content = reasoning.strip()
            if content:
                return content, None
            errors.append(f"{model_id}: empty response")
        except Exception as e:
            errors.append(f"{model_id}: {e}")

    return None, " | ".join(errors)

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

page = st.radio(
    "Navigate",
    ["💬 Chat with Gaia", "📊 Live Eco-Tracker"],
    horizontal=True,
    label_visibility="collapsed",
    key="page_nav",
)
st.divider()

# ==========================================
# PAGE: CHAT
# (st.chat_input must stay at the top level, not nested inside a tab/container,
# or Streamlit won't pin it to the bottom of the screen — that was the bug.)
# ==========================================
if page == "💬 Chat with Gaia":
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

    with st.expander("📸 Not sure what it's made of? Upload a photo and Gaia will identify it"):
        uploaded_image = st.file_uploader(
            "Upload a photo of the item",
            type=["jpg", "jpeg", "png"],
            key="material_photo",
        )
        if uploaded_image is not None:
            st.image(uploaded_image, width=250)
            if st.button("🔍 Identify this item", key="identify_btn"):
                with st.spinner("Gaia is taking a look..."):
                    mime = uploaded_image.type or "image/jpeg"
                    description, error = identify_material_from_image(uploaded_image.getvalue(), mime)

                if description:
                    user_message = f"I have this item: {description}. What should I do with it?"
                    st.session_state.chat_history.append({"role": "user", "content": user_message})
                    clean_hist = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ]
                    with st.spinner("Gaia is typing..."):
                        reply = respond(user_message, clean_hist)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()
                else:
                    st.error(f"Gaia couldn't identify that photo: {error}")

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
# PAGE: LIVE ECO-TRACKER
# ==========================================
else:
    st.write("### Claim your Eco-Points here as you complete Gaia's recommendations!")

    score = st.session_state.eco_score

    LEVELS = [
        (700, "🏆 Level 6: Planet Protector"),
        (500, "🌍 Level 5: Eco Hero"),
        (350, "🌳 Level 4: Earth Guardian"),
        (200, "🌿 Level 3: Green Sprout"),
        (75, "🌾 Level 2: Growing Green"),
        (0, "🌱 Level 1: Eco Seedling"),
    ]
    for i, (threshold, name) in enumerate(LEVELS):
        if score >= threshold:
            badge = name
            next_threshold = LEVELS[i - 1][0] if i > 0 else threshold
            floor = threshold
            progress = 1.0 if next_threshold == floor else (score - floor) / (next_threshold - floor)
            break
    progress = min(max(progress, 0.0), 1.0)

    col_stats, col_btns = st.columns(2)

    with col_stats:
        st.metric(label="Your Total Eco-Points", value=f"{score} pts")
        st.progress(progress, text="Progress to Next Earth Badge")
        st.info(f"Your Current Status: **{badge}**")
        st.caption(f"🔥 {len(st.session_state.eco_logs)} activities logged so far")

    with col_btns:
        st.write("#### What did you complete today?")

        def log_activity(label, points):
            st.session_state.eco_score += points
            st.session_state.eco_logs.append({
                "activity": label,
                "points": points,
                "timestamp": datetime.now(),
            })

        if st.button("✂️ Completed a DIY Craft (+25 pts)", use_container_width=True, key="craft_btn"):
            log_activity("Completed a DIY Craft", 25)
            st.rerun()

        if st.button("🗑️ Disposed of Materials Safely (+15 pts)", use_container_width=True, key="dispose_btn"):
            log_activity("Disposed of Materials Safely", 15)
            st.rerun()

        if st.button("🏷️ Listed/Sold an Item (+20 pts)", use_container_width=True, key="sell_btn"):
            log_activity("Listed/Sold an Item", 20)
            st.rerun()

        if st.button("🎁 Donated Used Clothing (+20 pts)", use_container_width=True, key="donate_btn"):
            log_activity("Donated Used Clothing", 20)
            st.rerun()

    st.write("### 📜 Your Green Activity Log")
    if st.session_state.eco_logs:
        log_df = pd.DataFrame(
            [
                {
                    "Date": entry["timestamp"].strftime("%b %d, %Y"),
                    "Time": entry["timestamp"].strftime("%I:%M %p"),
                    "Activity": entry["activity"],
                    "Points": f"+{entry['points']}",
                }
                for entry in reversed(st.session_state.eco_logs)
            ]
        )
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No activities logged yet. Start completing tasks!")
