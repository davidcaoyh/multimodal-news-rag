"""
Day 4: the demo. Per plan_7day.md this single screen *is* the project.

    streamlit run app/streamlit_app.py

What it shows, and why each piece is here
-----------------------------------------
The whole project is one comparison — text-only RAG (B1) vs multimodal RAG (M) —
so the default view is both arms side by side, running the *same* query through
the *same* generator with retrieval as the only difference. Single-arm views are
available in the sidebar; B0 (no retrieval) is opt-in as the hallucination ceiling.

Three things this screen has to make visible, because they are the result:

1. **The independent variable is alive.** M's retrieved set differs from B1's on
   ~53% of articles (Day 3 measurement). Articles only M found are badged, so a
   viewer sees the fusion actually changing what reaches the prompt.
2. **What actually went into the prompt.** The "prompt sent to the model" expander
   shows the literal evidence string. B1 and M share byte-identical instructions
   (D7 Rule 1); only the evidence block differs, and in M it carries the
   `[IMAGE k: "caption"]` lines. Images are shown in both columns because both
   arms retrieve articles that have images — but only M sends the caption to the
   model, and the expander is what proves it.
3. **Abstention has two distinct causes.** `abstained=True` is the tau gate and
   costs no API call; a summary of INSUFFICIENT_EVIDENCE with `abstained=False` is
   the model refusing on its own. Conflating them is how a broken 450-row run
   passed every check on Day 3, so they get different colours here.

Implementation notes
--------------------
- `summarize()` retrieves internally but returns text only, so the display calls
  `retrieve()` a second time for `Hit.image_path` and the passage text. Retrieval
  is deterministic and costs 21 ms, so the two calls agree by construction; the
  alternative is threading display state through the Day 5 interface file.
- Everything heavy is behind `@st.cache_resource` (encoders + FAISS, ~3 s once)
  or `@st.cache_data` (per-query results). Without those every widget interaction
  would pay the encoder load.
- Suggested chips use queries from the frozen set at default k/alpha, so they hit
  `data/llm_cache/` and cost $0. Moving the sliders makes a genuinely new call.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# src/ modules resolve data paths relative to cwd and import each other as a
# package. `streamlit run app/...` puts app/ on sys.path with cwd wherever the
# user launched it, so pin both to the repo root before importing anything.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.generate import ABSTAIN, TAU, summarize  # noqa: E402
from src.retrieve import retrieve  # noqa: E402

st.set_page_config(page_title="Multimodal RAG — news summarization",
                   page_icon="📰", layout="wide")

# Real queries come from the frozen set (cached -> free); the last chip is
# deliberately off-topic. Every test query is by construction about a real event
# in this corpus, so the tau gate essentially never fires on one (2/150 B1,
# 5/150 M) — without an off-topic chip the guardrail looks unimplemented.
CHIPS = [
    ("Nuclear power", "UK government plans nuclear power expansion"),
    ("Post Office", "Ed Davey claims Post Office misled him"),
    ("Stock Exchange", "arrests over plot to disrupt London Stock Exchange"),
    ("Off-topic ⚠", "how do I bake sourdough bread at home"),
]

ARM = {
    "B1": ("Text-only", "B1", "SBERT passages → FAISS"),
    "M": ("Multimodal", "M", "SBERT + CLIP → late fusion"),
    "B0": ("No retrieval", "B0", "the hallucination ceiling"),
}


# ---------------------------------------------------------------- loading

@st.cache_resource(show_spinner="Loading encoders and FAISS indexes…")
def warm_up():
    """Load indexes and force the first (slow) encoder pass out of the way.

    The first retrieve() in a fresh process pays the SBERT + CLIP model load;
    every one after is ~21 ms. Doing it here means the first real query the
    viewer types is fast.
    """
    retrieve("warm up the encoders", k=1)
    return True


@st.cache_data(show_spinner=False)
def run_arm(query: str, config: str, k: int, alpha: float, tau: float):
    """One (query, config) -> (record, hits). Cached per widget state."""
    rec = summarize(query, config=config, k=k, alpha=alpha, tau=tau)
    hits = [] if config == "B0" else retrieve(
        query, mode=("text" if config == "B1" else "multimodal"),
        k=k, alpha=alpha, include_test=False)
    return rec, hits


def arm_state(rec: dict) -> str:
    """ok | gated | refused — the three UI states, which are NOT two.

    `abstained` is the tau gate only. The model can return INSUFFICIENT_EVIDENCE
    itself with abstained=False, so the summary text has to be checked too.
    """
    if rec["abstained"]:
        return "gated"
    if rec["summary"].strip() == ABSTAIN:
        return "refused"
    return "ok"


# ---------------------------------------------------------------- rendering

def render_summary(rec: dict):
    state = arm_state(rec)
    if state == "gated":
        st.error(
            f"**ABSTAINED — retrieval guardrail**\n\n"
            f"Best raw text score over the top-{rec['n_hits']} hits is "
            f"`{rec['top_s_text']:.3f}`, below τ = `{TAU}`. The evidence is too weak "
            f"to ground a summary, so **no LLM call was made**.",
            icon="🛑")
    elif state == "refused":
        st.warning(
            f"**Model returned INSUFFICIENT_EVIDENCE**\n\n"
            f"Confidence `{rec['top_s_text']:.3f}` cleared τ = `{TAU}`, so the guardrail "
            f"did not fire — the generator judged the retrieved coverage to be about a "
            f"different subject and declined on its own.",
            icon="⚠️")
    else:
        st.success(rec["summary"], icon="✅")


def render_evidence(hits, config: str, unique_ids: set):
    """The retrieved articles. `unique_ids` are badged as found by this arm only."""
    if not hits:
        st.caption("No retrieval — B0 answers from the model's own knowledge.")
        return

    with_captions = config == "M"
    st.caption(
        f"{len(hits)} articles · "
        + ("passages **+ image captions** go into the prompt" if with_captions
           else "passages only — images shown for provenance, **not** sent to the model"))

    for i, h in enumerate(hits, 1):
        img, txt = st.columns([1, 3], vertical_alignment="top")
        with img:
            path = ROOT / h.image_path
            if path.exists():
                st.image(str(path), width="stretch")
        with txt:
            # Named rather than "only this arm": side by side, both columns badge
            # their own exclusives at once, and an unnamed badge reads as a
            # contradiction. `only M` is the fusion changing what reaches the prompt.
            badge = f" &nbsp;`only {config}`" if h.article_id in unique_ids else ""
            st.markdown(f"**{i}. {h.headline}**{badge}", unsafe_allow_html=True)
            st.caption(
                f"fused `{h.score:.3f}` · text `{h.s_text:.3f}` · image `{h.s_img:.3f}`")
            if with_captions:
                st.caption(f"🖼 *{h.caption}*")
        with st.expander(f"passages [{i}]"):
            for p in h.passages:
                st.write(p)
        st.divider()


def render_arm(query: str, config: str, k: int, alpha: float, tau: float,
               unique_ids: set):
    name, tag, how = ARM[config]
    st.subheader(f"{name} · `{tag}`")
    st.caption(how)

    rec, hits = run_arm(query, config, k, alpha, tau)
    render_summary(rec)

    if config != "B0" and hits:
        # TWO different statistics, because one number cannot do both jobs.
        #
        # gate score = max s_text over the top-k. This is the tau gate (D9) and the
        # only reason it is on screen is to explain the abstention state. It is a
        # BAD arm-comparison number: B1 ranks purely by s_text, so its max is the
        # pool-wide maximum, and M matches it whenever that same article survives
        # fusion — identical in both arms on 73% of the 150 test queries. Shown
        # side by side it looks frozen, which is what it did before this comment.
        #
        # mean text sim = mean s_text over the retrieved set. It moves with alpha and
        # separates the arms on 97% of queries — but it is NOT a quality score, and
        # must not be labelled as one. B1 selects the k highest-s_text articles, so its
        # mean is the maximum achievable over ANY k-subset and M is <= B1 by identity
        # (measured: 145 lower, 5 tied, 0 higher out of 150). Calling it "evidence
        # quality" makes the multimodal arm look worse by construction. Read the gap as
        # how far image fusion moved the set, not as which arm retrieved better.
        mean_text = sum(h.s_text for h in hits) / len(hits)
        a, b = st.columns(2)
        a.metric("gate score", f"{rec['top_s_text']:.3f}",
                 delta=f"{rec['top_s_text'] - tau:+.3f} vs τ", delta_color="normal",
                 help="max raw s_text over the top-k — the statistic the τ abstention "
                      "gate tests. Often identical in both arms: B1's is the pool-wide "
                      "maximum, and M matches it whenever that article survives fusion.")
        b.metric("mean text sim", f"{mean_text:.3f}",
                 help="mean raw s_text across the retrieved set. B1 maximises this by "
                      "construction, so M is never higher (0/150 test queries). The gap "
                      "measures how far image fusion moved the set — it is NOT evidence "
                      "that either arm retrieved better.")
        st.caption(f"image stream: max `{rec['top_s_img']:.3f}` · mean "
                   f"`{sum(h.s_img for h in hits) / len(hits):.3f}` — a fusion input, "
                   f"never the gate")

    # Label has to track the state: when the tau gate fires, the evidence block was
    # assembled but generate() was never called, so calling this "the prompt sent to
    # the model" would misreport what happened.
    gated = arm_state(rec) == "gated"
    with st.expander("evidence assembled — no prompt was sent" if gated
                     else "prompt sent to the model"):
        if rec["evidence"]:
            if gated:
                st.caption("Shown for inspection only. The guardrail fired before the "
                           "LLM call, so this text never reached the model.")
            st.code(rec["evidence"], language=None, wrap_lines=True)
        else:
            st.caption("B0 gets no evidence block — that is the point of the arm.")

    st.markdown("**Retrieved evidence**")
    render_evidence(hits, config, unique_ids)
    return rec


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Controls")
view = st.sidebar.radio(
    "View",
    ["Side by side (B1 vs M)", "Text-only (B1)", "Multimodal (M)"],
    help="The toggle. B1 and M run identical code with retrieval as the only difference.")
show_b0 = st.sidebar.checkbox(
    "Also show B0 (no retrieval)", value=False,
    help="LLM alone, no evidence — the hallucination upper bound.")

st.sidebar.divider()
k = st.sidebar.slider("k — articles retrieved", 1, 10, 5)
alpha = st.sidebar.slider(
    "α — text weight in fusion", 0.0, 1.0, 0.5, 0.05,
    help="score = α·s_text + (1−α)·s_img on per-query min-max normalized streams. "
         "α = 1.0 makes M identical to B1.")
tau = st.sidebar.slider(
    "τ — abstention threshold", 0.0, 0.8, TAU, 0.01,
    help="Abstain if max raw s_text over the top-k falls below τ. Raw cosine, never "
         "the fused score, which is min-max'd per query and so is relative.")
if alpha == 1.0:
    st.sidebar.info("α = 1.0 — M is now the same retrieval as B1.", icon="ℹ️")

st.sidebar.divider()
st.sidebar.caption(
    "SBERT `all-MiniLM-L6-v2` (384-d) · CLIP `ViT-B-32-quickgelu` (512-d) · "
    "FAISS `IndexFlatIP` · generator `gpt-4o-mini` @ temperature 0, held constant "
    "across all arms.\n\nPool: 873 BBC articles (Jan 2024), 11,052 passages, "
    "1,023 images. The 150 test articles are excluded from retrieval.")

if not os.path.exists(ROOT / ".env"):
    st.sidebar.error("No `.env` found — retrieval will work, generation will not.")


# ---------------------------------------------------------------- main

st.title("Does multimodal retrieval reduce hallucination?")
st.caption(
    "Same corpus, same generator, same prompt, same temperature. The **only** difference "
    "between the two arms is whether image similarity contributes to retrieval.")

warm_up()

st.markdown("**Try one:**")
chip_cols = st.columns(len(CHIPS))
for col, (label, q) in zip(chip_cols, CHIPS):
    if col.button(label, width="stretch"):
        st.session_state["query"] = q      # fills the box (widget not yet created)
        st.session_state["active"] = q

with st.form("search", clear_on_submit=False):
    st.text_input("Search the news corpus", key="query",
                  placeholder="e.g. UK government plans nuclear power expansion")
    submitted = st.form_submit_button("Summarize", type="primary")
if submitted:
    st.session_state["active"] = st.session_state.get("query", "").strip()

# The searched query is held in session state rather than derived from a one-shot
# "was the button just pressed" flag. Streamlit reruns the whole script on EVERY
# widget interaction, so with a one-shot flag moving the alpha or k slider would
# clear the results instead of recomputing them — which is exactly the interaction
# the sliders exist for.
query = st.session_state.get("active", "").strip()

if not query:
    st.info("Enter a query or pick a suggestion above. The off-topic chip demonstrates "
            "the abstention guardrail — real news queries almost never trip it.", icon="👆")
    st.stop()

st.divider()

# Which articles each arm found that the other did not. Computed before rendering
# so both columns can badge them — this is the independent variable made visible.
ids = {}
for c in ("B1", "M"):
    _, hits = run_arm(query, c, k, alpha, tau)
    ids[c] = [h.article_id for h in hits]
only = {"B1": set(ids["B1"]) - set(ids["M"]), "M": set(ids["M"]) - set(ids["B1"]),
        "B0": set()}

if view.startswith("Side"):
    n_new = len(only["M"])
    st.metric("articles multimodal retrieved that text-only did not",
              f"{n_new}/{len(ids['M'])}")
    if n_new == 0:
        st.caption("Identical top-k on this query — it happened on 5/150 test queries.")

    # Both on-screen numbers have a bias a viewer will otherwise misread: the gate
    # score looks frozen (identical on 73% of queries), and mean text sim is bounded
    # above by B1 by construction, which makes M look strictly worse. State both
    # limits rather than let the demo imply a result it cannot support.
    st.caption(
        "**Reading these numbers.** `gate score` is the τ statistic; it is identical "
        "in both arms on 73% of test queries, because B1's is the pool-wide maximum "
        "and M matches it whenever that article survives fusion. `mean text sim` is "
        "bounded above by B1 **by construction** — B1 selects the highest-`s_text` "
        "articles, so M is never higher (0/150). Neither number says which arm is "
        "better: that needs ground truth outside both streams — recall of the withheld "
        "gold article, or the faithfulness judge.")

    left, right = st.columns(2, gap="large")
    with left:
        render_arm(query, "B1", k, alpha, tau, only["B1"])
    with right:
        render_arm(query, "M", k, alpha, tau, only["M"])
else:
    config = "B1" if "B1" in view else "M"
    render_arm(query, config, k, alpha, tau, only[config])

if show_b0:
    st.divider()
    st.subheader("Baseline — no retrieval")
    st.caption("Nothing grounds this summary. It is the hallucination upper bound the "
               "retrieval arms are measured against.")
    render_arm(query, "B0", k, alpha, tau, set())
