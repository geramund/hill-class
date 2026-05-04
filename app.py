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

# ── Word lists ────────────────────────────────────────────────────────────────
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

NOUNS = [
    "man", "woman", "child", "house", "tree", "river", "stone", "door",
    "road", "sky", "fire", "wind", "hand", "eye", "light", "day", "night",
    "year", "place", "story", "voice", "water", "city", "field", "shadow",
    "bridge", "window", "floor", "wall", "ship", "mountain", "forest", "garden",
]

ADJECTIVES = [
    "cold", "dark", "small", "large", "old", "young", "long", "short",
    "still", "slow", "fast", "hard", "soft", "bright", "high", "low",
    "deep", "close", "open", "real", "black", "white", "heavy", "empty",
    "pale", "thin", "wide", "hollow", "broken", "worn", "bare", "sharp",
    "silent", "distant", "sudden", "strange", "familiar", "lost", "hidden",
    "gentle", "violent", "bitter", "sweet", "faint", "dense", "ancient",
    "narrow", "vast", "raw", "clean", "cloudy", "steady", "restless",
    "tired", "hungry", "frightened", "lonely", "curious", "furious", "numb",
]

VERBS = [
    "run", "walk", "see", "hear", "speak", "turn", "move", "fall", "rise",
    "hold", "keep", "find", "lose", "give", "take", "make", "leave", "stay",
    "begin", "end", "think", "feel", "watch", "break", "stand", "wait",
    "search", "call", "open", "reach", "follow", "hide", "pass", "pull",
    "carry", "drop", "climb", "drift", "sink", "burn", "freeze", "shake",
    "whisper", "shout", "cry", "laugh", "dream", "forget", "remember", "know",
    "enter", "escape", "return", "vanish", "appear", "gather", "scatter", "press",
    "grasp", "release", "strike", "catch", "throw", "push", "kneel", "crawl",
]

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
            deletions  = current_row[j] + 1
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

def visual_swap(word):
    clean = re.sub(r'[^a-zA-Z]', '', word)
    suffix = word[len(clean):]
    confusion = find_confusion(clean)
    if confusion:
        if clean and clean[0].isupper():
            confusion = confusion.capitalize()
        return confusion + suffix
    return word

def pos_swap(words, pos_prefix, replacement_list):
    """Return (index, original, replacement) for the last word whose POS tag
    starts with pos_prefix, swapped to a random entry from replacement_list."""
    tagged = nltk.pos_tag(words)
    for i in range(len(tagged) - 1, -1, -1):
        word, tag = tagged[i]
        if tag.startswith(pos_prefix):
            clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            candidates = [w for w in replacement_list if w != clean]
            if not candidates:
                continue
            replacement = random.choice(candidates)
            suffix = word[len(re.sub(r'[^a-zA-Z]', '', word)):]
            if word and word[0].isupper():
                replacement = replacement.capitalize()
            return i, word, replacement + suffix
    return None, None, None

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

MODELS = {
    "Llama 3.1 8B (fast)":   "llama-3.1-8b-instant",
    "Llama 3.3 70B":          "llama-3.3-70b-versatile",
    "Qwen3 32B":              "qwen/qwen3-32b",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a text continuation engine. Continue the text naturally "
    "with normal English words separated by spaces. Never merge words "
    "together. Never repeat the same phrase. Just write the next few "
    "words of the story."
)

def generate_n_words(context, n_words, temperature, client, system_prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Continue with the next {n_words} words:\n\n{context}"},
        ],
        max_tokens=n_words * 4,
        temperature=temperature,
        frequency_penalty=0.5,
    )
    output = completion.choices[0].message.content or ""
    output = clean_output(output.strip())
    output = remove_adverbs(output)

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

    return " ".join(truncated[:n_words])

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Autoregressive Feedback",
    page_icon="🔀",
    layout="centered",
)

st.markdown("""
<style>
:root { --primary-color: #888888 !important; }
div.stButton > button[kind="primary"] {
    background-color: #555555 !important;
    border-color: #555555 !important;
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #777777 !important;
    border-color: #777777 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #888888 !important;
    border-color: #888888 !important;
    box-shadow: #888888 0px 0px 0px 4px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="presentation"] > div > div {
    background-color: #888888 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="value" i],
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Value"] {
    color: #aaaaaa !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Adversarial Autoregressive Feedback")
st.caption(
    "A language model writes a story — but every few words, "
    "a word is silently swapped into its context. "
    "Watch meaning drift."
)

with st.expander("How does it work?"):
    st.markdown(
        """
        1. The model generates text in small chunks.
        2. At the end of each chunk, a word is silently replaced according to the swap mode.
        3. The model sees the swapped word as if it wrote it, then continues from there.
        4. Over time, meaning drifts — sometimes subtly, sometimes dramatically.
        5. Use **Pause / Inject** to stop at any time and add your own words before the model continues.

        **Gray italic words** = swapped substitutions (original shown above in small text) &nbsp;|&nbsp;
        **Dimmed words** = nudges &nbsp;|&nbsp;
        *Dark italic words* = your injections
        """
    )

st.divider()

# ── Session state init ────────────────────────────────────────────────────────
_defaults = {
    "gen_state":        "idle",   # idle | running | paused | done
    "context":          "",
    "display_parts":    [],
    "plain_text":       "",
    "words_generated":  0,
    "temperature_val":  0.7,
    "total_words_val":  150,
    "swap_every_val":   6,
    "system_prompt_val": DEFAULT_SYSTEM_PROMPT,
    "swap_mode_val":    "Visual similarity",
    "model_val":        "Llama 3.1 8B (fast)",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

is_running = st.session_state.gen_state == "running"

# ── Controls ──────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Starting prompt",
    value="From swerve of shore to bend of bay",
    height=100,
    disabled=is_running,
)

with st.expander("Edit system prompt"):
    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
        label_visibility="collapsed",
        help="Instructions given to the model. Try changing genre, voice, or language.",
        disabled=is_running,
    )

col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
        help="Higher = more unpredictable. Lower = more controlled.",
        disabled=is_running,
    )
with col2:
    total_words = st.slider(
        "Words to generate", min_value=50, max_value=400, value=150, step=25,
        help="Total words the model will write.",
        disabled=is_running,
    )
with col3:
    swap_every = st.slider(
        "Words before each swap", min_value=2, max_value=20, value=6, step=1,
        help="Lower = more frequent disruption.",
        disabled=is_running,
    )

swap_mode = st.radio(
    "Swap mode",
    ["Visual similarity", "Noun swap", "Adjective swap", "Verb swap"],
    horizontal=True,
    help=(
        "Visual similarity: swap with a look-alike word. "
        "Noun/Adjective/Verb: replace the last word of that type with a random one."
    ),
    disabled=is_running,
)

model_label = st.selectbox(
    "Model",
    list(MODELS.keys()),
    disabled=is_running,
)

run_button = st.button(
    "Generate", type="primary", use_container_width=True, disabled=is_running,
)

# ── Kick off generation ───────────────────────────────────────────────────────
if run_button:
    client = get_client()
    if client is None:
        st.error("API key not configured — please contact your instructor.")
        st.stop()
    if not prompt.strip():
        st.warning("Please enter a starting prompt.")
        st.stop()

    st.session_state.gen_state        = "running"
    st.session_state.context          = prompt.strip()
    st.session_state.display_parts    = [f'<span style="color:#999999">{prompt.strip()}</span>']
    st.session_state.plain_text       = prompt.strip()
    st.session_state.words_generated  = 0
    st.session_state.temperature_val  = temperature
    st.session_state.total_words_val  = total_words
    st.session_state.swap_every_val   = swap_every
    st.session_state.system_prompt_val = system_prompt
    st.session_state.swap_mode_val    = swap_mode
    st.session_state.model_val        = model_label
    st.rerun()

# ── Generation display ────────────────────────────────────────────────────────
if st.session_state.gen_state != "idle":
    st.divider()
    output_box = st.empty()

    def render():
        html = (
            '<p style="font-family: Georgia, serif; font-size: 17px; '
            'line-height: 2.8; word-wrap: break-word; color: #000000;">'
            + " ".join(st.session_state.display_parts)
            + "</p>"
        )
        output_box.markdown(html, unsafe_allow_html=True)

    render()

    # Download button always visible once generation has started
    ss = st.session_state
    export_text = (
        f"=== Settings ===\n"
        f"Model:              {ss.model_val}\n"
        f"Swap mode:          {ss.swap_mode_val}\n"
        f"Temperature:        {ss.temperature_val}\n"
        f"Words to generate:  {ss.total_words_val}\n"
        f"Words before swap:  {ss.swap_every_val}\n"
        f"System prompt:      {ss.system_prompt_val}\n"
        f"\n=== Story ===\n"
        f"{ss.plain_text}\n"
    )
    st.download_button(
        "Save as text",
        data=export_text,
        file_name="story.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # ── Paused: inject UI ─────────────────────────────────────────────────────
    if st.session_state.gen_state == "paused":
        inject_text = st.text_input(
            "inject",
            key="inject_input",
            placeholder="Type your continuation here, then click Continue…",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Continue", use_container_width=True, type="primary"):
                if inject_text.strip():
                    st.session_state.display_parts.append(
                        f'<span style="color:#444444; font-style:italic">{inject_text.strip()}</span>'
                    )
                    st.session_state.context    += " " + inject_text.strip()
                    st.session_state.plain_text += " " + inject_text.strip()
                st.session_state.gen_state = "running"
                st.rerun()
        with c2:
            if st.button("Stop", use_container_width=True):
                st.session_state.gen_state = "done"
                st.rerun()

    # ── Running: generate next chunk ─────────────────────────────────────────
    elif st.session_state.gen_state == "running":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Pause / Inject", use_container_width=True):
                st.session_state.gen_state = "paused"
                st.rerun()
        with c2:
            if st.button("Stop", use_container_width=True, key="stop_running"):
                st.session_state.gen_state = "done"
                st.rerun()

        wg    = st.session_state.words_generated
        total = st.session_state.total_words_val

        if wg < total:
            st.progress(min(wg / total, 1.0), text=f"{wg} / {total} words")
            client = get_client()
            try:
                chunk = generate_n_words(
                    st.session_state.context,
                    st.session_state.swap_every_val,
                    st.session_state.temperature_val,
                    client,
                    st.session_state.system_prompt_val,
                    MODELS[st.session_state.model_val],
                )
                words = chunk.split()

                if not words or len(words) < 2:
                    nudge = random.choice(NUDGES)
                    st.session_state.display_parts.append(
                        f'<span style="color:#777777">{nudge}</span>'
                    )
                    st.session_state.context    += " " + nudge
                    st.session_state.plain_text += " " + nudge
                else:
                    mode = st.session_state.swap_mode_val

                    if mode == "Visual similarity":
                        for word in words[:-1]:
                            st.session_state.display_parts.append(word)
                            st.session_state.context    += " " + word
                            st.session_state.plain_text += " " + word

                        last_word = words[-1]
                        swapped   = visual_swap(last_word)
                        if swapped != last_word:
                            st.session_state.display_parts.append(
                                f'<ruby style="color:#888888; font-style:italic">{swapped}'
                                f'<rt style="font-size:0.65em; color:#aaaaaa; font-style:normal">{last_word}</rt></ruby>'
                            )
                            st.session_state.context    += " " + swapped
                            st.session_state.plain_text += f" [{last_word}→{swapped}]"
                        else:
                            st.session_state.display_parts.append(last_word)
                            st.session_state.context    += " " + last_word
                            st.session_state.plain_text += " " + last_word

                    else:
                        pos_map = {
                            "Noun swap":      ("NN", NOUNS),
                            "Adjective swap": ("JJ", ADJECTIVES),
                            "Verb swap":      ("VB", VERBS),
                        }
                        prefix, repl_list = pos_map[mode]
                        swap_idx, original, replacement = pos_swap(words, prefix, repl_list)

                        for i, word in enumerate(words):
                            if i == swap_idx and swap_idx is not None:
                                st.session_state.display_parts.append(
                                    f'<ruby style="color:#888888; font-style:italic">{replacement}'
                                    f'<rt style="font-size:0.65em; color:#aaaaaa; font-style:normal">{original}</rt></ruby>'
                                )
                                st.session_state.context    += " " + replacement
                                st.session_state.plain_text += f" [{original}→{replacement}]"
                            else:
                                st.session_state.display_parts.append(word)
                                st.session_state.context    += " " + word
                                st.session_state.plain_text += " " + word

                    st.session_state.words_generated += len(words)

                render()
                st.rerun()

            except Exception as e:
                st.error(f"Generation error: {e}")
                st.session_state.gen_state = "done"
        else:
            st.session_state.gen_state = "done"
            st.rerun()
