import json
import os
import html
import io
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pypdf import PdfReader
from docx import Document
from PIL import Image


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


SAMPLE_RESUME = """
Jane Candidate
Backend Engineer | Python, AWS, SQL, Docker

Summary
Backend engineer with 5+ years building REST APIs and data-driven services.
Strong experience with Python/FastAPI, PostgreSQL, AWS Lambda, and Docker-based deployments.

Experience
Backend Developer, Acme Analytics (2021 - Present)
- Built REST APIs using FastAPI and deployed them with AWS Lambda and API Gateway.
- Wrote and optimized PostgreSQL queries (joins, indexing strategies, query performance tuning).
- Containerized services with Docker; improved deployment consistency across environments.
- Implemented CI/CD pipelines (GitHub Actions) and added unit/integration tests for critical endpoints.

Skills
Python, FastAPI, Flask, PostgreSQL, SQL, REST APIs, AWS Lambda, S3, API Gateway, Docker, CI/CD

Projects
- API service for reporting dashboards (FastAPI + PostgreSQL + AWS)
- ETL jobs for transforming raw events into analytics tables
""".strip()


def render_logo(logo_path: str, width: int = 180) -> None:
    """
    Display a left-aligned logo with white background removed and tight-cropped.
    Falls back to the raw image if processing fails.
    """
    try:
        with Image.open(logo_path) as im:
            im = im.convert("RGBA")
            pixels = im.getdata()
            new_pixels = []
            # Remove near-white "tile" pixels more conservatively:
            # - require very high brightness
            # - and require the color channels to be close (grey/white)
            # This avoids deleting light parts of the logo itself.
            WHITE_CUTOFF = 250
            GREY_TOL = 8
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

            bbox = im.getbbox()
            if bbox:
                im = im.crop(bbox)

            buf = io.BytesIO()
            im.save(buf, format="PNG")
            st.image(buf.getvalue(), width=width)
            return
    except Exception:
        pass

    st.image(logo_path, width=width)


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return api_key


def get_openai_client() -> OpenAI:
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file."
        )
    return OpenAI(api_key=api_key)


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


def call_jobfit_agent(job_description: str, resume: str) -> Dict[str, Any]:
    client = get_openai_client()
    prompt = build_prompt(job_description, resume)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON-generating assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError("Unexpected OpenAI API response format.") from e

    if not content:
        raise RuntimeError("OpenAI returned an empty response.")

    # In case the model accidentally includes markdown fences, strip them.
    content_stripped = content.strip()
    if content_stripped.startswith("```"):
        # Remove leading/trailing code fences
        content_stripped = content_stripped.strip("`")
        # Try to remove possible language tag like json\n at the start
        if "\n" in content_stripped:
            first_line, rest = content_stripped.split("\n", 1)
            if first_line.lower() in {"json", "javascript", "ts"}:
                content_stripped = rest

    try:
        parsed = json.loads(content_stripped)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from model response: {e}\nRaw content:\n{content}") from e

    return parsed


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
        return ""


def extract_text_from_docx(file) -> str:
    try:
        doc = Document(file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception:
        return ""


def fetch_text_from_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def get_text_input_block(
    label: str,
    placeholder: str,
    sample_text: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Composite input allowing:
    - Direct text paste
    - File upload (PDF/DOCX/TXT)
    - URL to fetch content from
    """
    st.markdown(f"#### {label}")
    state_key = f"{label.lower().replace(' ', '_')}_text"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    if sample_text:
        if st.button(f"Use Sample {label}", key=f"{state_key}_sample", type="secondary"):
            st.session_state[state_key] = sample_text
            st.rerun()

    tabs = st.tabs(["Text", "File / URL"])

    text_value: str = ""
    has_content = False

    with tabs[0]:
        text_value = st.text_area(
            f"{label} (Text)",
            placeholder=placeholder,
            height=260,
            key=state_key,
        )
        if text_value.strip():
            has_content = True

    with tabs[1]:
        uploaded_file = st.file_uploader(
            f"{label} file (PDF / Word / TXT)",
            type=["pdf", "docx", "txt"],
            key=f"{label.lower().replace(' ', '_')}_file",
        )
        url_input = st.text_input(
            f"{label} URL",
            placeholder="Paste a link to the JD/Resume (PDF, DOCX, or HTML page)...",
            key=f"{label.lower().replace(' ', '_')}_url",
        )

        extracted_text: str = ""

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(
                ".pdf"
            ):
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

        if not extracted_text and url_input.strip():
            fetched = fetch_text_from_url(url_input)
            extracted_text = fetched

        if extracted_text:
            has_content = True
            st.info("Content loaded successfully.")
            # Prefer extracted text when text area is empty
            if not text_value.strip():
                text_value = extracted_text

    return text_value.strip(), has_content


def main() -> None:
    st.set_page_config(
        page_title="JobFit Agent",
        page_icon="📄",
        layout="wide",
    )

    # App theming
    st.markdown(
        """
        <style>
          /* Background that adapts to Streamlit light/dark themes */
          .stApp {
            --jobfit-bg-top: #f6fbf7;
            --jobfit-bg-mid: #f8fafc;
            --jobfit-bg-bottom: #ffffff;

            background:
              radial-gradient(1200px circle at 15% 10%, rgba(16,185,129,0.16) 0%, rgba(16,185,129,0) 45%),
              radial-gradient(900px circle at 85% 20%, rgba(59,130,246,0.14) 0%, rgba(59,130,246,0) 50%),
              linear-gradient(180deg, var(--jobfit-bg-top) 0%, var(--jobfit-bg-mid) 60%, var(--jobfit-bg-bottom) 100%);
          }

          /* When Streamlit dark mode is active */
          html[data-theme="dark"] .stApp,
          body[data-theme="dark"] .stApp,
          .stApp[data-theme="dark"] {
            --jobfit-bg-top: #0b1220;
            --jobfit-bg-mid: #0f172a;
            --jobfit-bg-bottom: #0b1220;
            color-scheme: dark;
          }

          /* Dark mode: ensure app text stays high-contrast */
          html[data-theme="dark"] .stApp,
          body[data-theme="dark"] .stApp,
          .stApp[data-theme="dark"] {
            color: #e5e7eb !important;
          }
          html[data-theme="dark"] .stApp h1,
          html[data-theme="dark"] .stApp h2,
          html[data-theme="dark"] .stApp h3,
          html[data-theme="dark"] .stApp h4,
          html[data-theme="dark"] .stApp h5,
          html[data-theme="dark"] .stApp p,
          html[data-theme="dark"] .stApp span,
          html[data-theme="dark"] .stApp label {
            color: #e5e7eb !important;
            opacity: 1 !important;
          }

          @media (prefers-color-scheme: dark) {
            .stApp {
              --jobfit-bg-top: #0b1220;
              --jobfit-bg-mid: #0f172a;
              --jobfit-bg-bottom: #0b1220;
              color-scheme: dark;
              color: #e5e7eb !important;
            }
          }

          /* Make Streamlit chrome blend with background */
          [data-testid="stHeader"], [data-testid="stToolbar"] {
            background: transparent !important;
          }

          /* Slightly soften blocks */
          div[data-testid="stVerticalBlock"] {
            border-radius: 14px;
          }
          div[data-testid="stMarkdownContainer"] {
            border-radius: 14px;
          }

          /* Dark mode: remove any white tile behind the logo image */
          [data-testid="stImage"] {
            background: transparent !important;
          }
          [data-testid="stImage"] > div {
            background: transparent !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([3, 1])

    with header_left:
        # App logo
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            render_logo(logo_path, width=180)

        st.title("JobFit Agent")
        st.caption("Analyze the job-resume fit")

    with header_right:
        api_key_present = bool(load_api_key())
        status_container = st.container()
        with status_container:
            if api_key_present:
                st.markdown(
                    "<div style='text-align: right; padding-top: 0.75rem;'>"
                    "<span style='background-color: #14532d; color: #bbf7d0; padding: 0.25rem 0.75rem;"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align: right; padding-top: 0.75rem;'>"
                    "<span style='background-color: #7f1d1d; color: #fecaca; padding: 0.25rem 0.75rem;"
                    "</span></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        job_description, has_jd = get_text_input_block(
            label="Job Description",
            placeholder="Paste the full job description here or use the File / URL tab...",
            sample_text=SAMPLE_JOB_DESCRIPTION,
        )

    with col2:
        resume, has_resume = get_text_input_block(
            label="Resume",
            placeholder="Paste the candidate's resume here or use the File / URL tab...",
            sample_text=SAMPLE_RESUME,
        )

    analyze_clicked = st.button("Analyze Fit", type="primary")

    if analyze_clicked:
        if not has_jd or not has_resume:
            st.error("Please provide both a Job Description and a Resume (via text, file, or URL) before analyzing.")
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
                cards_html = "".join(
                    [
                        (
                            "<div style='background-color:#dcfce7;color:#14532d;"
                            "padding:8px 10px;border-radius:12px;margin:4px;"
                            "display:inline-block;font-size:0.92rem;line-height:1.2;'>"
                            f"{html.escape(skill)}"
                            "</div>"
                        )
                        for skill in skills
                    ]
                )
                st.markdown(
                    f"<div style='display:flex;flex-wrap:wrap;'>{cards_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("No clear overlapping skills or keywords detected.")

            st.markdown("### Missing Skills")
            if result["missing_skills"]:
                skills = sorted(set(result["missing_skills"]))
                cards_html = "".join(
                    [
                        (
                            "<div style='background-color:#fee2e2;color:#7f1d1d;"
                            "padding:8px 10px;border-radius:12px;margin:4px;"
                            "display:inline-block;font-size:0.92rem;line-height:1.2;'>"
                            f"{html.escape(skill)}"
                            "</div>"
                        )
                        for skill in skills
                    ]
                )
                st.markdown(
                    f"<div style='display:flex;flex-wrap:wrap;'>{cards_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("No obvious missing skills or keywords detected.")

        st.divider()

        st.markdown("### Resume Improvement Suggestions")
        for idx, suggestion in enumerate(result["resume_improvements"], start=1):
            if suggestion.strip():
                st.markdown(f"- **Suggestion {idx}**: {suggestion}")


if __name__ == "__main__":
    main()

