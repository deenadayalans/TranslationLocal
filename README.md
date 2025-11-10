## Local LLM Translator (Ollama + Streamlit)

Translate text and files locally using your Ollama models. No data leaves your machine.

### Features
- Auto-detect source language with manual override
- Translate to any target language
- Text translation with streaming output
- File translation: `.txt`, `.md`, `.docx`, `.pdf`
  - Outputs `.txt` or `.docx` (PDF layout not preserved; text content translated)
- Select any locally installed Ollama model
- In-app model management (pull models) and A/B comparison tab

### Prerequisites
- Python 3.9+
- Ollama installed and running (`ollama serve`), default host: `http://localhost:11434`

### Recommended models
Instruction-tuned multilingual models tend to translate better:
- `qwen2.5:7b-instruct` (good multilingual support)
- `llama3.1:8b` or `llama3.2:3b-instruct`
- `mistral:7b-instruct`

Pull a model (examples):

```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
```

### Setup

```bash
cd /Users/srinivasand/Desktop/Repo/TranslationLocal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your Ollama host is non-default, set:

```bash
export OLLAMA_HOST=http://localhost:11434
```

### Run

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (typically `http://localhost:8501`).

### Using the app
- Translate Text: paste text, auto-detect source or select, pick target, and translate. Toggle streaming in sidebar.
- Translate File: upload `.txt`, `.md`, `.docx`, or `.pdf` and download as `.txt` or `.docx`.
- Compare Models: choose two installed models and see translations side-by-side on the same input.
- Models: view installed models, quick-pull recommended ones, or enter any name from the Ollama Library.

### Notes
- PDF translation extracts text only; complex layouts/images are not preserved. Consider uploading the original `.docx` when available for better structure retention.
- For large files, the app chunks content to avoid model context limits.
- Language detection uses `langdetect`; you can override manually if detection is off.
- If no models appear in the sidebar, pull one with `ollama pull <model>` and restart the app.

### Troubleshooting
- Ensure Ollama is running: `ollama serve` (or its background service).
- Check connectivity: `curl http://localhost:11434/api/tags` should list models.
- If the app says a Python package is missing, re-run `pip install -r requirements.txt`.


