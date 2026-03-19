# JobFit Agent

JobFit Agent is a simple web app built with Python and Streamlit that uses the OpenAI GPT‑4o model to evaluate how well a resume matches a given job description.

The app:

- **Computes a match score** out of 100 between the Job Description (JD) and the Resume.
- **Lists matching skills** found in both the JD and the Resume.
- **Lists missing skills/keywords** that appear in the JD but not clearly in the Resume.
- **Provides 3 tailored resume improvement suggestions** for the specific JD.
- **Gives a final verdict**: `"Strong Apply"`, `"Apply with Modifications"`, or `"Skip"`.

All analysis is done through the OpenAI API, and the model is instructed to return results in a strict JSON format which the app then parses and displays in a clean, structured UI.

## Requirements

- Python 3.9+ (recommended)
- An OpenAI API key with access to the GPT‑4o model

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

4. **Set up your OpenAI API key**

   - Copy `.env.example` to `.env`:

     ```bash
     copy .env.example .env  # On Windows (PowerShell: cp .env.example .env)
     # cp .env.example .env  # On macOS / Linux
     ```

   - Edit `.env` and set your key:

     ```text
     OPENAI_API_KEY=sk-your-real-key-here
     ```

   The app uses `python-dotenv` to load `OPENAI_API_KEY` from `.env` at runtime.

## Running the App

From inside the project directory:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

## Using the JobFit Agent

1. Paste the **Job Description** into the left text area.
2. Paste the **Resume** into the right text area.
3. Click **“Analyze Fit”**.

You will see:

- **Match score** and **verdict** at the top.
- **Matching skills** and **missing skills** in two columns.
- **Three concrete resume improvement suggestions**.
- An optional expandable section showing the **raw JSON** response from the OpenAI API.

If the OpenAI API key is missing, or if the API call fails for any reason, the app will display a clear error message instead of crashing.

## Notes

- The app uses the `gpt-4o` model via the `openai` Python library.
- The `.env` file is ignored by Git via `.gitignore` to keep your API key private.

