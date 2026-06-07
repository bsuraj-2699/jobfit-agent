import json
import logging
import os
import html
import io
import base64
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pypdf import PdfReader
from docx import Document
from PIL import Image


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("jobfit-agent")

# --- Configuration -----------------------------------------------------------
# Guardrails to keep API cost, latency, and request size bounded.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "30000"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
# Generous enough to cover "thinking" models (e.g. Gemini 2.5 Flash) that spend
# part of the token budget on internal reasoning before emitting the answer.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# --- Multi-provider support --------------------------------------------------
# Whichever API key is present in the environment selects the provider. Most
# providers expose an OpenAI-compatible Chat Completions endpoint, so they share
# one client (just a different base_url). Anthropic (Claude) is the exception:
# it uses the official `anthropic` SDK, not an OpenAI-compatibility shim.
#
# A best-fit default model is chosen per provider for this JSON evaluation task;
# override any of them with the listed env var, or force a global model with
# LLM_MODEL and a specific provider with LLM_PROVIDER.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "kind": "openai",
        "env_keys": ["OPENAI_API_KEY"],
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o",
        "base_url": None,
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "kind": "anthropic",
        "env_keys": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "model_env": "ANTHROPIC_MODEL",
        "default_model": "claude-opus-4-8",
        "base_url": None,
    },
    "gemini": {
        "label": "Google Gemini",
        "kind": "openai",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "groq": {
        "label": "Groq",
        "kind": "openai",
        "env_keys": ["GROQ_API_KEY"],
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "mistral": {
        "label": "Mistral",
        "kind": "openai",
        "env_keys": ["MISTRAL_API_KEY"],
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
    },
}

# Detection order when no LLM_PROVIDER is forced (first key found wins).
PROVIDER_ORDER = ["openai", "anthropic", "gemini", "groq", "mistral"]

# JSON schema for Anthropic structured outputs (guarantees parseable JSON).
JOBFIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "match_score": {"type": "number"},
        "matching_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "resume_improvements": {"type": "array", "items": {"type": "string"}},
        "verdict": {
            "type": "string",
            "enum": ["Strong Apply", "Apply with Modifications", "Skip"],
        },
    },
    "required": [
        "match_score",
        "matching_skills",
        "missing_skills",
        "resume_improvements",
        "verdict",
    ],
    "additionalProperties": False,
}


def detect_provider() -> tuple[Optional[str], Optional[str]]:
    """Return (provider_name, api_key) based on env vars, or (None, None).

    Honors an explicit LLM_PROVIDER override; otherwise picks the first
    provider in PROVIDER_ORDER whose API key is set.
    """
    load_dotenv()

    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced:
        if forced not in PROVIDERS:
            logger.warning("Unknown LLM_PROVIDER=%r; ignoring.", forced)
        else:
            for env_key in PROVIDERS[forced]["env_keys"]:
                value = os.getenv(env_key, "").strip()
                if value:
                    return forced, value
            return forced, None  # forced but key missing — surfaced to user

    for name in PROVIDER_ORDER:
        for env_key in PROVIDERS[name]["env_keys"]:
            value = os.getenv(env_key, "").strip()
            if value:
                return name, value
    return None, None


def resolve_model(provider: str) -> str:
    """Pick the model for a provider: LLM_MODEL > <PROVIDER>_MODEL > default."""
    override = os.getenv("LLM_MODEL", "").strip()
    if override:
        return override
    spec = PROVIDERS[provider]
    per_provider = os.getenv(spec["model_env"], "").strip()
    if per_provider:
        return per_provider
    return spec["default_model"]


SAMPLE_JOB_DESCRIPTION = """
Software Engineer (Backend) - Job Description

We are looking for a backend engineer to build and maintain REST APIs and data pipelines.

Responsibilities
- Develop Python services and backend REST endpoints (FastAPI / Flask)
- Write efficient SQL for PostgreSQL and optimize queries
- Deploy and operate services on AWS (Lambda / S3 / API Gateway)
- Containerize applications using Docker and support CI/CD workflows

Preferred Qualifications
- Experience integrating with React frontends
- Familiarity with Kubernetes and IaC (Terraform)
- Strong testing practices (unit/integration)

Key Skills
Python, REST APIs, FastAPI, SQL, PostgreSQL, AWS, Docker, CI/CD, React, Kubernetes, Terraform
""".strip()


def render_logo(logo_path: str, width: int = 120, align: str = "left") -> None:
    """
    Render a transparent, tight-cropped logo.

    We embed the processed image directly in HTML to avoid Streamlit's image
    wrapper introducing a "white tile" background in some themes.
    """

    align = align.lower().strip()
    if align not in {"left", "center"}:
        align = "left"
    margin_style = "margin: 0;" if align == "left" else "margin: 0 auto;"

    try:
        with Image.open(logo_path) as im:
            im = im.convert("RGBA")
            pixels = im.getdata()
            new_pixels = []
            # Remove near-white "tile" pixels conservatively:
            # - require very high brightness
            # - and require the color channels to be close (grey/white)
            # This avoids deleting light parts of the logo itself.
            WHITE_CUTOFF = 245
            GREY_TOL = 15
            for r, g, b, a in pixels:
                if a == 0:
                    new_pixels.append((r, g, b, a))
                elif r >= WHITE_CUTOFF and g >= WHITE_CUTOFF and b >= WHITE_CUTOFF and (
                    abs(r - g) <= GREY_TOL and abs(g - b) <= GREY_TOL
                ):
                    new_pixels.append((r, g, b, 0))
                else:
                    new_pixels.append((r, g, b, a))
            im.putdata(new_pixels)

            # Tight-crop to the non-transparent content.
            bbox = im.getbbox()
            if bbox:
                im = im.crop(bbox)

            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        st.markdown(
            (
                "<img "
                f"src='data:image/png;base64,{b64}' "
                f"style='width:{width}px; height:auto; background: transparent; display:block; {margin_style}' "
                "/>"
            ),
            unsafe_allow_html=True,
        )
        return
    except Exception:
        pass

    # Fallback: embed the raw image without processing.
    try:
        with open(logo_path, "rb") as f:
            png_bytes = f.read()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        st.markdown(
            (
                "<img "
                f"src='data:image/{os.path.splitext(logo_path)[1].lstrip('.').lower()};base64,{b64}' "
                f"style='width:{width}px; height:auto; display:block; {margin_style}' "
                "/>"
            ),
            unsafe_allow_html=True,
        )
    except Exception:
        st.image(logo_path, width=width)


SYSTEM_PROMPT = "You are a precise JSON-generating assistant."


def build_prompt(job_description: str, resume: str) -> str:
    return f"""
You are a Job Fit Evaluation Agent. Carefully analyze the following Job Description (JD) and Resume.

JOB DESCRIPTION:
\"\"\"{job_description}\"\"\"

RESUME:
\"\"\"{resume}\"\"\"

Return ONLY a valid JSON object with this exact structure and keys:
{{
  "match_score": <number between 0 and 100>,
  "matching_skills": [
    "<string 1>",
    "<string 2>",
    "<string 3>",
    .
    .
    .
    "<string n>"
  ],
  "missing_skills": [
    "<string 1>",
    "<string 2>",
    "<string 3>",
    .
    .
    .
    "<string n>"
  ],
  "resume_improvements": [
    <string 1>,
    <string 2>,
    <string 3>,
    <string 4>,
    .
    .
    .
    <string n>
  ],
  "verdict": <one of "Strong Apply", "Apply with Modifications", "Skip">
}}

Rules:
- Do not include any commentary outside the JSON.
- The JSON must be syntactically valid and parsable by standard JSON libraries.
- The match_score must be conservative and use the full 0–100 range.
- Scoring rubric (be strict and realistic):
  - 90–100: Nearly perfect fit. Resume explicitly covers almost all critical responsibilities, tech stack, requirements, preferred qualifications, seniority/industry expectations, and relevant certifications in the JD with strong evidence (projects, outcomes, years of experience, education).
  - 75–89: Strong fit. Most core skills, required qualifications, and responsibilities are present with clear evidence, but a few notable gaps exist in secondary skills, domain exposure, or credentials.
  - 55–74: Partial fit. Some relevant overlap in primary skills and responsibilities, but multiple important gaps in skills, requirements, preferred experience, domain, or level; would require meaningful upskilling, additional certifications, or resume tailoring to align better.
  - 30–54: Weak fit. Only limited overlap in either skills, experience level, education, certifications, or responsibilities; resume is not clearly targeting this JD and lacks evidence for several parameters.
  - 0–29: Very poor fit. Almost no meaningful alignment to the JD requirements, seniority/industry needs, critical technologies, or certifications.
- "matching_skills" should be skills/keywords clearly present in both JD and Resume.
- "missing_skills" should be important skills/keywords present in the JD but clearly missing or very weak in the Resume.
- "resume_improvements" must contain exactly 3 concrete, actionable suggestions tailored to this JD.
- Choose "verdict" based on overall alignment:
  - "Strong Apply": Very high alignment, minimal gaps.
  - "Apply with Modifications": Decent alignment but some notable gaps; improvements needed.
  - "Skip": Significant mismatch or major gaps.
"""


def _truncate(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    """Clamp very long inputs to keep token usage and cost bounded."""
    if len(text) <= limit:
        return text
    logger.warning("Input truncated from %d to %d characters.", len(text), limit)
    return text[:limit]


def _parse_json_response(content: Optional[str]) -> Dict[str, Any]:
    """Parse a model's text response into JSON, tolerating markdown fences."""
    if not content:
        raise RuntimeError("The model returned an empty response.")

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            first_line, rest = stripped.split("\n", 1)
            if first_line.lower() in {"json", "javascript", "ts"}:
                stripped = rest

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from model response: {e}\nRaw content:\n{content}"
        ) from e


def _call_openai_compatible(
    provider: str, api_key: str, model: str, prompt: str
) -> Dict[str, Any]:
    """Call any OpenAI-compatible Chat Completions endpoint (OpenAI/Gemini/Groq/Mistral)."""
    base_url = PROVIDERS[provider]["base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    def _create(use_json: bool):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "timeout": REQUEST_TIMEOUT_SECONDS,
        }
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        response = _create(use_json=True)
    except OpenAIError:
        # Some providers/models reject response_format — retry without JSON mode.
        logger.warning("%s rejected JSON mode; retrying without it.", provider)
        try:
            response = _create(use_json=False)
        except OpenAIError as e:
            logger.exception("%s API call failed.", provider)
            raise RuntimeError(f"{PROVIDERS[provider]['label']} API call failed: {e}") from e

    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError("Unexpected API response format.") from e

    # If the model hit the token cap, the JSON is cut off mid-string. Give a
    # clear, actionable error instead of a confusing JSON parse failure.
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            "The model's response was cut off (token limit reached) before it "
            "could finish the JSON. Try again, or increase MAX_OUTPUT_TOKENS in "
            "your .env (some models spend many tokens on internal reasoning)."
        )

    return _parse_json_response(content)


def _call_anthropic(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call Claude via the official Anthropic SDK with structured JSON output."""
    import anthropic  # imported lazily so the dep is only needed for the Claude path

    client = anthropic.Anthropic(api_key=api_key)
    # Note: Opus 4.x rejects temperature/top_p/budget_tokens — do not pass them.
    base_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }

    try:
        # Structured outputs guarantee schema-valid JSON on supported models.
        response = client.messages.create(
            **base_kwargs,
            output_config={"format": {"type": "json_schema", "schema": JOBFIT_SCHEMA}},
        )
    except (anthropic.APIError, TypeError) as first_error:
        # Fall back to a plain call (older SDKs / models without structured outputs).
        logger.warning("Anthropic structured output unavailable (%s); falling back.", first_error)
        try:
            response = client.messages.create(**base_kwargs)
        except anthropic.APIError as e:
            logger.exception("Anthropic API call failed.")
            raise RuntimeError(f"Anthropic Claude API call failed: {e}") from e

    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            "The model's response was cut off (token limit reached) before it "
            "could finish the JSON. Try again, or increase MAX_OUTPUT_TOKENS in your .env."
        )

    content = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        None,
    )
    return _parse_json_response(content)


def call_jobfit_agent(job_description: str, resume: str) -> Dict[str, Any]:
    provider, api_key = detect_provider()
    if not provider:
        raise RuntimeError(
            "No API key found. Set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "GEMINI_API_KEY, GROQ_API_KEY, or MISTRAL_API_KEY in your .env file."
        )
    if not api_key:
        raise RuntimeError(
            f"LLM_PROVIDER is set to '{provider}', but its API key is not set. "
            f"Set {PROVIDERS[provider]['env_keys'][0]} in your .env file."
        )

    model = resolve_model(provider)
    logger.info("Using provider=%s model=%s", provider, model)
    prompt = build_prompt(_truncate(job_description), _truncate(resume))

    if PROVIDERS[provider]["kind"] == "anthropic":
        return _call_anthropic(api_key, model, prompt)
    return _call_openai_compatible(provider, api_key, model, prompt)


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    # Provide safe defaults if any key is missing or mis-typed
    validated: Dict[str, Any] = {}

    match_score = result.get("match_score")
    try:
        match_score = float(match_score)
    except (TypeError, ValueError):
        match_score = 0.0
    match_score = max(0.0, min(100.0, match_score))
    validated["match_score"] = match_score

    def ensure_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(v) for v in value if isinstance(v, (str, int, float))]

    validated["matching_skills"] = ensure_str_list(result.get("matching_skills"))
    validated["missing_skills"] = ensure_str_list(result.get("missing_skills"))

    resume_improvements = ensure_str_list(result.get("resume_improvements"))
    if len(resume_improvements) < 3:
        resume_improvements += [""] * (3 - len(resume_improvements))
    validated["resume_improvements"] = resume_improvements[:3]

    verdict = str(result.get("verdict", "")).strip()
    allowed_verdicts = {"Strong Apply", "Apply with Modifications", "Skip"}
    if verdict not in allowed_verdicts:
        # Fallback based on score
        if match_score >= 80:
            verdict = "Strong Apply"
        elif match_score >= 50:
            verdict = "Apply with Modifications"
        else:
            verdict = "Skip"
    validated["verdict"] = verdict

    return validated


def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    except Exception:
        logger.exception("Failed to extract text from PDF.")
        return ""


def extract_text_from_docx(file) -> str:
    try:
        doc = Document(file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception:
        logger.exception("Failed to extract text from DOCX.")
        return ""


def get_job_description_input() -> tuple[str, bool]:
    """Job Description input: paste-only text area."""
    label = "Job Description"
    st.markdown(f"#### {label}")
    state_key = "job_description_text"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    if st.button(f"Use Sample {label}", key=f"{state_key}_sample", type="secondary"):
        st.session_state[state_key] = SAMPLE_JOB_DESCRIPTION
        st.rerun()

    text_value = st.text_area(
        f"{label} (Text)",
        placeholder="Paste the full job description here...",
        height=320,
        key=state_key,
    )

    return text_value.strip(), bool(text_value.strip())


def get_resume_input() -> tuple[str, bool]:
    """Resume input: file upload only (PDF / Word / TXT)."""
    label = "Resume"
    st.markdown(f"#### {label}")

    uploaded_file = st.file_uploader(
        f"{label} file (PDF / Word / TXT)",
        type=["pdf", "docx", "txt"],
        key="resume_file",
    )

    if uploaded_file is None:
        return "", False

    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ) or uploaded_file.name.lower().endswith(".docx"):
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        # Fallback for plain text
        try:
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            extracted_text = ""

    extracted_text = extracted_text.strip()
    if not extracted_text:
        st.warning(
            "Could not extract any text from the uploaded file. "
            "It may be empty, image-only/scanned, or corrupted."
        )
        return "", False

    st.info("Resume loaded successfully.")
    return extracted_text, True


def main() -> None:
    st.set_page_config(
        page_title="JobFit Agent",
        page_icon="logo.png",
        layout="wide",
    )

    header_left, header_right = st.columns([3, 1])

    with header_left:
        # App logo
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            render_logo(logo_path, width=120, align="left")

        st.title("JobFit Agent")
        st.caption("Analyze the job-resume fit")

    with header_right:
        provider, api_key = detect_provider()
        status_container = st.container()
        with status_container:
            # Let Streamlit apply theme-appropriate colors.
            _, status_col = st.columns([3, 1])
            with status_col:
                if provider and api_key:
                    st.success(
                        f"{PROVIDERS[provider]['label']}\n\n`{resolve_model(provider)}`"
                    )
                elif provider:
                    st.error(f"{PROVIDERS[provider]['label']} key missing")
                else:
                    st.error("No API key found")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        job_description, has_jd = get_job_description_input()

    with col2:
        resume, has_resume = get_resume_input()

    analyze_clicked = st.button("Analyze Fit", type="primary")

    if analyze_clicked:
        if not has_jd or not has_resume:
            st.error("Please paste a Job Description and upload a Resume file before analyzing.")
            return

        with st.spinner("Analyzing...."):
            try:
                raw_result = call_jobfit_agent(job_description, resume)
                result = validate_result(raw_result)
            except RuntimeError as e:
                st.error(str(e))
                return
            except Exception as e:  # Fallback for unexpected issues
                st.error(f"An unexpected error occurred: {e}")
                return

        st.subheader("Match Overview")
        score_col, verdict_col = st.columns(2)
        with score_col:
            match_score = float(result["match_score"])
            st.progress(match_score / 100.0, text=f"Match Score: {match_score:.1f} / 100")
        with verdict_col:
            verdict_badge = result["verdict"]
            if verdict_badge == "Strong Apply":
                st.success(f"Verdict: {verdict_badge}")
            elif verdict_badge == "Apply with Modifications":
                st.warning(f"Verdict: {verdict_badge}")
            else:
                st.error(f"Verdict: {verdict_badge}")

        st.divider()

        # Both matching and missing skills stacked on the left side
        skills_col, _ = st.columns([2, 1])

        with skills_col:
            st.markdown("### Matching Skills")
            if result["matching_skills"]:
                skills = sorted(set(result["matching_skills"]))
                bullets = "\n".join([f"- {html.escape(skill)}" for skill in skills])
                st.markdown(bullets)
            else:
                st.write("No clear overlapping skills or keywords detected.")

            st.markdown("### Missing Skills")
            if result["missing_skills"]:
                skills = sorted(set(result["missing_skills"]))
                bullets = "\n".join([f"- {html.escape(skill)}" for skill in skills])
                st.markdown(bullets)
            else:
                st.write("No obvious missing skills or keywords detected.")

        st.divider()

        st.markdown("### Resume Improvement Suggestions")
        has_suggestion = False
        for idx, suggestion in enumerate(result["resume_improvements"], start=1):
            if suggestion.strip():
                has_suggestion = True
                st.markdown(f"- **Suggestion {idx}**: {suggestion}")
        if not has_suggestion:
            st.write("No improvement suggestions were generated.")

        with st.expander("Raw JSON response"):
            st.json(result)


if __name__ == "__main__":
    main()

