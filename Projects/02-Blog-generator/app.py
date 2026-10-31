from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="BlogCraft AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        color: #f0f6fc;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .feature-card h4 {
        color: #58a6ff;
        margin-bottom: 8px;
    }

    .feature-card p {
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 0;
    }

    .stButton > button {
        width: 100%;
        background: #238636;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #2ea043;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: #090d12;
        border-right: 1px solid #30363d;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #161b22;
        color: #f0f6fc;
    }

    div[data-baseweb="select"] > div {
        background: #161b22;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

MODEL = os.getenv("MODEL")

if not MODEL:
    st.error("MODEL is missing. Add it to your .env file.")
    st.stop()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Check your .env file.")
    st.stop()


@st.cache_resource
def load_model():
    return ChatGroq(
        model=MODEL,
        temperature=0.5,
        max_tokens=2500
    )


llm = load_model()
parser = StrOutputParser()


# ==========================================
# 4. HUMAN WRITING STYLE
# ==========================================

HUMAN_STYLE = """
Write naturally and clearly.

Rules:
- Use simple, natural language.
- Avoid robotic and repetitive sentences.
- Vary sentence length.
- Prefer active voice.
- Explain difficult concepts using relevant examples.
- Avoid unnecessary jargon and filler.
- Avoid generic introductions and conclusions.
- Keep paragraphs short and readable.
- Do not invent statistics, facts, or quotations.
- Make every paragraph useful to the reader.
"""


# ==========================================
# 5. OUTLINE PROMPT
# ==========================================

OUTLINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional blog strategist.

        Create a SHORT, practical outline for a blog post.

        Writing style:
        {human_style}

        Requirements:
        - Create one compelling title.
        - Include a short introduction.
        - Use 3-5 main headings for a typical short blog.
        - Add subsections only when necessary.
        - Include relevant examples where useful.
        - End with a concise conclusion.
        - Add FAQs only when specifically useful.
        - Avoid repetitive or unnecessary sections.

        IMPORTANT:
        The outline must match the requested word count.
        Do not create too many sections.
        Return only the outline, not the complete blog.
        """
    ),
    (
        "human",
        """
        Topic: {topic}
        Tone: {tone}
        Audience: {audience}
        Target word count: {word_count}
        Additional instructions: {instructions}

        Create the outline.
        """
    )
])


# ==========================================
# 6. DRAFT PROMPT
# ==========================================

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a concise and engaging blog writer.

        Write the blog using the supplied outline.

        Writing style:
        {human_style}

        Rules:
        - Target approximately {word_count} words.
        - Stay within 15% of the target word count.
        - Treat the word count as a strict constraint.
        - Follow the supplied outline.
        - Use Markdown headings.
        - Keep paragraphs short.
        - Explain the main ideas clearly.
        - Include examples only when they add value.
        - Avoid unnecessary background information.
        - Do not repeat the same points.
        - Avoid lengthy introductions and conclusions.
        - Do not invent facts or citations.
        - Do not add sections absent from the outline
          unless they are necessary.
        - Prioritize useful information over filler.

        Return only the complete blog draft in Markdown.
        """
    ),
    (
        "human",
        """
        Topic: {topic}
        Tone: {tone}
        Audience: {audience}
        Target word count: {word_count}
        Additional instructions: {instructions}

        Blog outline:
        {outline}

        Write a concise blog post following these requirements.
        """
    )
])


# ==========================================
# 7. POLISH PROMPT
# ==========================================

POLISH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional blog editor.

        Polish the supplied blog draft without changing
        its main meaning.

        Writing style:
        {human_style}

        Rules:
        - Keep the blog close to {word_count} words.
        - Preserve the original structure and key information.
        - Correct grammar, spelling, and punctuation.
        - Improve clarity and readability.
        - Remove repetition, filler, and unnecessary sentences.
        - Keep the selected tone consistent.
        - Preserve useful examples.
        - Do not add unsupported facts or citations.
        - Do not expand the article unnecessarily.
        - Use clean Markdown formatting.

        IMPORTANT:
        Do not make the article longer just to improve it.
        Prefer concise writing.

        Return only the final polished blog in Markdown.
        """
    ),
    (
        "human",
        """
        Topic: {topic}
        Tone: {tone}
        Audience: {audience}
        Target word count: {word_count}
        Additional instructions: {instructions}

        Draft:
        {draft}

        Polish the draft while keeping it concise.
        """
    )
])


# ==========================================
# 8. LANGCHAIN CHAINS
# ==========================================

outline_chain = OUTLINE_PROMPT | llm | parser

draft_chain = DRAFT_PROMPT | llm | parser

final_chain = POLISH_PROMPT | llm | parser


# ==========================================
# 9. BLOG GENERATION FUNCTION
# ==========================================

def generate_blog(
    topic,
    tone,
    audience,
    word_count,
    instructions,
    progress=None
):
    if not topic.strip():
        raise ValueError("Please enter a blog topic.")

    common_inputs = {
        "topic": topic,
        "tone": tone,
        "audience": audience,
        "word_count": word_count,
        "instructions": instructions or "None",
        "human_style": HUMAN_STYLE
    }

    # Stage 1: Generate outline
    if progress:
        progress.write("📝 Creating the blog outline...")

    outline = outline_chain.invoke(common_inputs)

    # Stage 2: Generate draft
    if progress:
        progress.write("✍️ Writing the first draft...")

    draft_inputs = {
        **common_inputs,
        "outline": outline
    }

    draft = draft_chain.invoke(draft_inputs)

    # Stage 3: Polish draft
    if progress:
        progress.write("✨ Polishing the final blog...")

    final_inputs = {
        **common_inputs,
        "draft": draft
    }

    final_blog = final_chain.invoke(final_inputs)

    return {
        "outline": outline,
        "draft": draft,
        "final_blog": final_blog
    }


# ==========================================
# 10. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## ✍️ BlogCraft AI")

    st.caption("Your AI-powered blog writing assistant")

    st.divider()

    st.markdown("### ⚙️ Configuration")

    st.markdown(f"**Model:** `{MODEL}`")

    st.markdown("**Pipeline**")
    st.caption("Outline → Draft → Polish")

    st.divider()

    st.markdown("### 💡 Writing Tips")

    st.caption(
        "Be specific about your topic. "
        "A focused topic usually produces a better blog."
    )

    st.caption(
        "Use additional instructions to request examples, "
        "code snippets, or a particular structure."
    )

    st.divider()

    if st.button("🗑️ Clear Results"):
        st.session_state.pop("blog_result", None)
        st.session_state.pop("blog_settings", None)
        st.rerun()

    st.caption("Built with Python, LangChain, Groq and Streamlit")


# ==========================================
# 11. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">✍️ BlogCraft AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Turn your ideas into clear, engaging blog posts.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 12. INPUT FORM
# ==========================================

st.markdown("### 📝 Create a new blog")

with st.form("blog_form"):

    topic = st.text_area(
        "Blog topic",
        placeholder=(
            "Example: How AI agents are changing "
            "software development"
        ),
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:
        tone = st.selectbox(
            "Writing tone",
            [
                "Conversational",
                "Professional",
                "Educational",
                "Casual",
                "Persuasive",
                "Technical"
            ]
        )

    with col2:
        audience = st.selectbox(
            "Target audience",
            [
                "General readers",
                "Beginners",
                "Students",
                "Developers",
                "Professionals",
                "Business owners"
            ]
        )

    word_count = st.select_slider(
        "Blog length",
        options=[300, 500, 800, 1000],
        value=500,
        format_func=lambda x: f"{x} words"
    )

    instructions = st.text_area(
        "Additional instructions (optional)",
        placeholder=(
            "Example: Include a Python example, "
            "use simple English, avoid unnecessary theory."
        ),
        height=80
    )

    generate_button = st.form_submit_button(
        "✨ Generate Blog",
        use_container_width=True
    )


# ==========================================
# 13. GENERATE BLOG
# ==========================================

if generate_button:

    if not topic.strip():
        st.warning("Please enter a topic first.")

    else:
        try:
            with st.status(
                "Generating your blog...",
                expanded=True
            ) as status:

                result = generate_blog(
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    word_count=word_count,
                    instructions=instructions,
                    progress=status
                )

                status.update(
                    label="Blog generated successfully!",
                    state="complete",
                    expanded=False
                )

            st.session_state.blog_result = result

            st.session_state.blog_settings = {
                "topic": topic,
                "tone": tone,
                "audience": audience,
                "word_count": word_count
            }

        except Exception as error:
            st.error("Something went wrong while generating the blog.")
            st.caption(str(error))


# ==========================================
# 14. DISPLAY RESULTS
# ==========================================

if "blog_result" in st.session_state:

    result = st.session_state.blog_result
    settings = st.session_state.blog_settings

    final_blog = result["final_blog"]

    # Approximate word count
    actual_word_count = len(final_blog.split())

    st.divider()

    st.markdown("## 📄 Your Generated Blog")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Words",
            actual_word_count,
            delta=f"Target: {settings['word_count']}"
        )

    with col2:
        st.metric("Tone", settings["tone"])

    with col3:
        st.metric("Audience", settings["audience"])

    tab_blog, tab_outline, tab_draft = st.tabs(
        ["📖 Final Blog", "📋 Outline", "📝 First Draft"]
    )

    with tab_blog:
        st.markdown(final_blog)

        st.download_button(
            label="⬇️ Download Blog (.md)",
            data=final_blog,
            file_name="generated_blog.md",
            mime="text/markdown",
            use_container_width=True
        )

    with tab_outline:
        st.markdown(result["outline"])

    with tab_draft:
        st.markdown(result["draft"])

    st.caption(
        "Review factual claims before publishing, especially "
        "technical details, statistics, and references."
    )

else:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Focused content</h4>
        <p>Choose your topic, tone, audience, and desired length.</p>
    </div>

    <div class="feature-card">
        <h4>🔗 Three-stage generation</h4>
        <p>Generate an outline, write a draft, and polish the result.</p>
    </div>

    <div class="feature-card">
        <h4>📥 Ready to export</h4>
        <p>Download your final blog as a Markdown file.</p>
    </div>
    """, unsafe_allow_html=True)