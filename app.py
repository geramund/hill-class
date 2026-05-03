import streamlit as st
import re
import random
import os
import nltk

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
@st.cache_resource
def bootstrap_nltk():
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)

bootstrap_nltk()

# ── Groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    from groq import Groq
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key)

# ── Word list ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_wordlist():
    common = """
    the and that have with from they been have their said each she which do how her
    time will way about many then them write would like so these her long make thing
    see him two has look more day could go come did number sound no most people over
    know water than call first who may down side been now find any new take get place
    made live where after back little only round man year came show every good me give
    our under name very through just form sentence great think say help low line differ
    turn cause much mean before move right boy old too same tell does set three want air
    well also play small end put home read hand port large spell add even land here must
    big high such follow act why ask men change went light kind off need house picture
    try us again animal point mother world near build self earth father head stand own
    page should country found answer school grow study still learn plant cover food sun
    four between state keep eye never last door between city tree cross farm hard start
    might story saw far sea draw left late run while press close night real life
    few north open seem together next white children begin got walk example ease paper
    often always music those both mark book letter until mile river car feet care second
    enough plain girl usual young ready above ever red list though feel talk bird soon
    body dog family direct pose leave song measure door product black short numeral class
    wind question happen complete ship area half rock order fire south problem piece told
    knew pass since top whole king space heard best hour better true during hundred five
    remember step early hold west ground interest reach fast verb sing listen six table
    travel less morning ten simple several vowel toward war lay against pattern slow center
    love person money serve appear road map rain rule govern pull cold notice voice unit
    power town fine drive study spoke break piece tell knew pass
    """.split()
    return set(w.strip().lower() for w in common if len(w.strip()) > 1)

WORDLIST = load_wordlist()

NUDGES = [
    "Suddenly,", "But then", "Meanwhile,", "Without warning,",
    "Far away,", "He remembered", "The sound of",
]

# ── Core logic ────────────────────────────────────────────────────────────────
def visual_distance(a, b):
    if len(a) < len(b):
        return visual_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def find_confusion(word):
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    if len(clean) < 3:
        return None
    candidates = []
    for w in WORDLIST:
        if w == clean:
            continue
        if abs(len(w) - len(clean)) > 2:
            continue
        dist = visual_distance(clean, w)
        if 1 <= dist <= 2:
            candidates.append((dist, w))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    best_dist = candidates[0][0]
    best = [w for d, w in candidates if d == best_dist]
    return random.choice(best)

def transform(word):
    clean = re.sub(r'[^a-zA-Z]', '', word)
    suffix = word[len(clean):]
    confusion = find_confusion(clean)
    if confusion:
        if len(clean) > 0 and clean[0].isupper():
            confusion = confusion.capitalize()
        return confusion + suffix
    return word

def clean_output(text):
    words = text.split()
    return " ".join(w for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) <= 20)

def remove_adverbs(text):
    words = text.split()
    if not words:
        return text
    tagged = nltk.pos_tag(words)
    return " ".join(word for word, tag in tagged if tag not in ('RB', 'RBR', 'RBS'))

def get_stem(w):
    w = re.sub(r'[^a-zA-Z]', '', w).lower()
    return w[:5] if len(w) >= 5 else w

DEFAULT_SYSTEM_PROMPT = (
    "You are a text continuation engine. Continue the text naturally "
    "with normal English words separated by spaces. Never merge words "
    "together. Never repeat the same phrase. Just write the next few "
    "words of the story."
)

def generate_n_words(context, n_words, temperature, client, system_prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Continue with the next {n_words} words:\n\n{context}",
            },
        ],
        max_tokens=n_words * 4,
        temperature=temperature,
        frequency_penalty=0.5,
    )
    output = completion.choices[0].message.content or ""
    output = clean_output(output.strip())
    output = remove_adverbs(output)

    # truncate before trigram repeat
    words_out = output.split()
    seen_trigrams = set()
    truncated = []
    for i, w in enumerate(words_out):
        if i >= 2:
            trigram = (words_out[i - 2], words_out[i - 1], w)
            if trigram in seen_trigrams:
                break
            seen_trigrams.add(trigram)
        truncated.append(w)
    output = " ".join(truncated)

    # truncate before stem loop
    words_out = output.split()
    truncated = []
    window = []
    for w in words_out:
        stem = get_stem(w)
        window.append(stem)
        if len(window) > 20:
            window.pop(0)
        if window.count(stem) >= 3:
            break
        truncated.append(w)

    words = truncated[:n_words]
    return " ".join(words)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Autoregressive Feedback",
    page_icon="🔀",
    layout="centered",
)

st.markdown("""
<style>
/* override Streamlit's primary accent (drives slider fill, thumb, value label) */
:root {
    --primary-color: #888888 !important;
}
/* primary button */
div.stButton > button[kind="primary"] {
    background-color: #555555 !important;
    border-color: #555555 !important;
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #777777 !important;
    border-color: #777777 !important;
}
/* slider thumb */
[data-testid="stSlider"] [role="slider"] {
    background-color: #888888 !important;
    border-color: #888888 !important;
    box-shadow: #888888 0px 0px 0px 4px !important;
}
/* slider filled track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="presentation"] > div > div {
    background-color: #888888 !important;
}
/* slider value label above thumb */
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="value" i],
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Value"] {
    color: #aaaaaa !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Adversarial Autoregressive Feedback")
st.caption(
    "A language model writes a story — but every few words, "
    "a similar-looking word is silently swapped into its context. "
    "Watch meaning drift."
)

with st.expander("How does it work?"):
    st.markdown(
        """
        1. The model generates text in small chunks.
        2. At the end of each chunk, the **last word is silently replaced** with a
           visually similar word (e.g. *love* → *live*, *light* → *night*).
        3. The model sees the swapped word as if it wrote it, then continues from there.
        4. Over time, meaning drifts — sometimes subtly, sometimes dramatically.

        **Gray italic words** = swapped substitutions (original shown above) &nbsp;|&nbsp; **Dimmed words** = nudges (injected when the model gets stuck)
        """
    )

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Starting prompt",
    value="From swerve of shore to bend of bay",
    height=100,
)

with st.expander("Edit system prompt"):
    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
        label_visibility="collapsed",
        help="Instructions given to the model before it starts writing. "
             "Try changing the genre, voice, or language.",
    )

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher = more unpredictable and creative. Lower = more controlled.",
    )

with col2:
    total_words = st.slider(
        "Words to generate",
        min_value=50,
        max_value=400,
        value=150,
        step=25,
        help="Total number of words the model will write.",
    )

with col3:
    swap_every = st.slider(
        "Words before each swap",
        min_value=2,
        max_value=20,
        value=6,
        step=1,
        help="How many words the model writes before a swap is injected. "
             "Lower = more frequent disruption.",
    )

run_button = st.button("Generate", type="primary", use_container_width=True)

# ── Generation ────────────────────────────────────────────────────────────────
if run_button:
    client = get_client()
    if client is None:
        st.error("API key not configured — please contact your instructor.")
        st.stop()

    if not prompt.strip():
        st.warning("Please enter a starting prompt.")
        st.stop()

    st.divider()
    output_box = st.empty()
    context = prompt.strip()
    words_generated = 0

    # prompt shown in grey, generated text in default colour
    display_parts = [f'<span style="color:#888888">{context}</span>']

    def render():
        html = (
            '<p style="font-family: Georgia, serif; font-size: 17px; '
            'line-height: 2.8; word-wrap: break-word; color: #ffffff;">'
            + " ".join(display_parts)
            + "</p>"
        )
        output_box.markdown(html, unsafe_allow_html=True)

    render()

    progress = st.progress(0, text="Starting…")

    try:
        while words_generated < total_words:
            chunk = generate_n_words(context, swap_every, temperature, client, system_prompt)
            words = chunk.split()

            if not words or len(words) < 2:
                nudge = random.choice(NUDGES)
                display_parts.append(
                    f'<span style="color:#777777">{nudge}</span>'
                )
                context += " " + nudge
                render()
                continue

            for word in words[:-1]:
                display_parts.append(word)
                context += " " + word
                words_generated += 1

            last_word = words[-1]
            swapped = transform(last_word)

            if swapped != last_word:
                display_parts.append(
                    f'<ruby style="color:#888888; font-style:italic">{swapped}'
                    f'<rt style="font-size:0.65em; color:#aaaaaa; font-style:normal">{last_word}</rt></ruby>'
                )
                context += " " + swapped

            else:
                display_parts.append(last_word)
                context += " " + last_word

            words_generated += len(words)
            render()
            pct = min(words_generated / total_words, 1.0)
            progress.progress(pct, text=f"{words_generated} / {total_words} words")

    except Exception as e:
        st.error(f"Generation error: {e}")

    progress.empty()
