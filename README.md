# JobFit Agent

JobFit Agent is a web app built with Python and Streamlit that uses an LLM to evaluate how well a resume matches a given job description. It supports **multiple providers** — OpenAI, Anthropic (Claude), Google Gemini, Groq, and Mistral — and automatically uses whichever provider's API key you set in `.env`, with a best-fit model chosen per provider.

The app:

- **Computes a match score** out of 100 between the Job Description (JD) and the Resume.
- **Lists matching skills** found in both the JD and the Resume.
- **Lists missing skills/keywords** that appear in the JD but not clearly in the Resume.
- **Provides 3 tailored resume improvement suggestions** for the specific JD.
- **Gives a final verdict**: `"Strong Apply"`, `"Apply with Modifications"`, or `"Skip"`.

The selected model is instructed to return results in strict JSON, which the app parses and displays in a clean, structured UI.

## Supported providers

Set **one** of the following API keys in `.env`. If several are set, detection order is OpenAI → Anthropic → Gemini → Groq → Mistral.

| Provider  | Env var             | Default model              |
|-----------|---------------------|----------------------------|
| OpenAI    | `OPENAI_API_KEY`    | `gpt-4o`                   |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-8`          |
| Gemini    | `GEMINI_API_KEY`    | `gemini-3.1-flash-lite`         |
| Groq      | `GROQ_API_KEY`      | `llama-3.3-70b-versatile`  |
| Mistral   | `MISTRAL_API_KEY`   | `mistral-large-latest`     |

Force a provider with `LLM_PROVIDER=<name>`, override the model globally with `LLM_MODEL`, or per provider with e.g. `OPENAI_MODEL` / `ANTHROPIC_MODEL`. See `.env.example` for all options.

## Configuration

All configuration is via environment variables (loaded from `.env`). Everything except the provider API key is optional.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` / `MISTRAL_API_KEY` | — | Provider API key. The first one found (in detection order) selects the provider. |
| `LLM_PROVIDER` | auto-detect | Force a specific provider (`openai`, `anthropic`, `gemini`, `groq`, `mistral`). |
| `LLM_MODEL` | per-provider default | Override the model for whichever provider is active. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GEMINI_MODEL` / `GROQ_MODEL` / `MISTRAL_MODEL` | see table above | Per-provider model override. |
| `MAX_OUTPUT_TOKENS` | `4096` | Max tokens the model may generate. Raise it if responses get cut off — "thinking" models (e.g. Gemini 2.5 Flash) spend part of the budget on internal reasoning. |
| `MAX_INPUT_CHARS` | `30000` | JD/resume text is truncated to this length before being sent, to bound cost and latency. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Per-request timeout for the LLM API call. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, ...). |

## Requirements

- Python 3.9+ (recommended)
- An API key for any one supported provider (see the table above)

## Setup

1. **Clone or create the project directory**

   Navigate into the `jobfit-agent` folder.

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your provider API key**

   - Copy `.env.example` to `.env`:

     ```bash
     copy .env.example .env  # On Windows (PowerShell: cp .env.example .env)
     # cp .env.example .env  # On macOS / Linux
     ```

   - Edit `.env` and set the key for the provider you want, e.g.:

     ```text
     OPENAI_API_KEY=sk-your-real-key-here
     # or ANTHROPIC_API_KEY=...   GEMINI_API_KEY=...   GROQ_API_KEY=...   MISTRAL_API_KEY=...
     ```

   The app uses `python-dotenv` to load the key from `.env` at runtime.

## Running the App

From inside the project directory:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

## Using the JobFit Agent

1. Paste the **Job Description** into the left text area (or click *Use Sample Job Description*).
2. Upload the **Resume** file (PDF, Word, or TXT) on the right.
3. Click **“Analyze Fit”**.

You will see:

- **Match score** and **verdict** at the top.
- **Matching skills** and **missing skills** in two columns.
- **Three concrete resume improvement suggestions**.
- An optional expandable section showing the **raw JSON** response from the model.

If no API key is set, or if the API call fails for any reason, the app will display a clear error message instead of crashing.

## Troubleshooting

- **"No API key found"** — Set one provider key in `.env` (see the table above) and restart the app.
- **"The model's response was cut off (token limit reached)"** — The model ran out of output budget. Increase `MAX_OUTPUT_TOKENS` in `.env`, or switch to a non-reasoning model (e.g. `GEMINI_MODEL=gemini-2.0-flash`). Reasoning models like Gemini 2.5 Flash consume tokens on internal thinking before answering.
- **"Could not extract any text from the uploaded file"** — The resume may be empty, image-only/scanned, or corrupted. Re-export it as a text-based PDF/DOCX or paste the text.
- **Wrong provider is used** — When multiple keys are set, detection order is OpenAI → Anthropic → Gemini → Groq → Mistral. Set `LLM_PROVIDER` to force a specific one.

## Notes

- OpenAI/Gemini/Groq/Mistral are called through the `openai` Python library (they expose OpenAI-compatible endpoints); Claude is called through the official `anthropic` library.
- The `.env` file is ignored by Git via `.gitignore` to keep your API key private.

