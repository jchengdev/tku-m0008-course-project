# 🏥 Taiwan International Student Healthcare & Admin Explainer (TKU M0008 Course Project)

A RAG-based AI assistant designed to provide accurate, context-grounded information on healthcare, National Health Insurance (NHI), ARC, and administrative procedures for international students in Taiwan. Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**, managed modernly using **`uv`**.

## 📂 Project Structure

```text
tku-m0008-course-project/
│
├── .venv/                 # Isolated Python 3.11 virtual environment (ignored by Git)
├── .streamlit/            # Streamlit settings & local secrets
│   └── secrets.toml       # Local API key storage (ignored by Git — never commit!)
├── .chroma_db/            # Local Chroma vector database cache (ignored by Git)
├── data/                  # Curated official PDF reference documents
├── .gitattributes         # Enforces LF line endings globally for cross-platform Git sync
├── .gitignore             # Excludes .venv, secrets.toml, vector DB cache, and OS junk
├── .python-version        # Pins Python 3.11 across machines
├── pyproject.toml         # Project definition & dependencies list (like package.json)
├── uv.lock                # Pinned dependency lockfile (like package-lock.json)
└── app.py                 # Main Streamlit + RAG application code
```

## 🚀 1. Clone the Repository & Configure Git

Open your terminal (Bash, WSL, or Windows PowerShell), clone the repository, and open it in VS Code:

```bash
git clone https://github.com/jchengdev/tku-m0008-course-project.git
cd tku-m0008-course-project
code .
```

### Cross-Platform Git Line Endings (`CRLF` vs. `LF`)

To prevent line-ending conflicts between Windows and Linux/WSL teammates, ensure `.gitattributes` and `.gitignore` are committed in the repository root and configure your local Git client:

- **Linux / WSL / macOS:**

  ```bash
  git config --global core.autocrlf input
  ```

- **Full Windows:**

  ```powershell
  git config --global core.autocrlf true
  ```

### Recommended VS Code Extensions

Install the following extensions for the best development experience:

- **Python** (`ms-python.python`)
- **Ruff** (`charliermarsh.ruff`)
- **Rainbow CSV** (`mechatroner.rainbow-csv`)

## 💻 2. System Preparation (Linux / Ubuntu / WSL)

If you are running on Ubuntu or WSL, update your system packages and install standard build headers before setting up Python tools:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y pkg-config python3-dev libdbus-1-dev libcairo2-dev libgirepository1.0-dev
```

*(Note: Keep your system updated periodically whenever needed using `sudo apt update && sudo apt upgrade`.)*

## 🐍 3. Environment Setup (`uv` + Python 3.11)

We use **`uv`** with **Python 3.11** across all operating systems to guarantee consistent dependency resolution and C-extension compatibility (ChromaDB, NumPy, etc.), avoiding compilation issues on bleeding-edge Python versions (e.g., Python 3.15).

### Step A: Install `uv`

- **Linux (Ubuntu / WSL) & macOS:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Full Windows (PowerShell):**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  *(Close and reopen PowerShell after installation so the updated `PATH` takes effect.)*

### Step B: Configure Python 3.11 & Virtual Environment

Run these commands inside the project root directory (works identically on Linux, WSL, and Windows PowerShell):

```bash
# Initialize uv workspace configuration (skip if pyproject.toml already exists)
uv init --no-package

# Install and pin Python 3.11
uv python install 3.11
uv python pin 3.11

# Create the isolated virtual environment
uv venv --python 3.11

# Verify active Python version (should output Python 3.11.x)
uv run python --version
```

### Step C: Install Project Dependencies

- **If syncing an existing repository (`pyproject.toml` and `uv.lock` are already present):**

  ```bash
  uv sync
  ```

- **If setting up dependencies from scratch:**

  ```bash
  uv add streamlit langchain langchain-community langchain-google-genai langchain-chroma langchain-text-splitters chromadb pypdf pandas scikit-learn numpy
  ```

## 🔑 4. API Key Configuration (Local Development)

To run the application locally, provide your Google Gemini API key by setting up a local secrets file:

1. Create the `.streamlit` directory in your project root:
   - **Linux / WSL / macOS:** `mkdir -p .streamlit`
   - **Windows (PowerShell):** `New-Item -ItemType Directory -Force -Path .streamlit`
2. Create `.streamlit/secrets.toml` and add your key:

   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
   ```

   *(⚠️ `.streamlit/secrets.toml` is excluded by `.gitignore` so your API key is never leaked to GitHub. You can also input it directly in the app sidebar at runtime if preferred).*

## ▶️ 5. Running the Application

1. **Confirm Reference Documents:**
   Ensure the `data/` folder exists and contains your curated official English PDF guides (e.g., Taiwan NHI guides, ARC handbooks) before running the app:
   - **Linux / WSL / macOS:** `mkdir -p data`
   - **Windows (PowerShell):** `New-Item -ItemType Directory -Force -Path data`

2. **Start the Streamlit Server:**
   Because `uv run` automatically routes execution through `.venv`, you do not need platform-specific activation scripts (`source .venv/bin/activate` vs. `.venv\Scripts\activate`):
   - **Linux (Desktop), macOS, & Full Windows:**

     ```bash
     uv run streamlit run app.py
     ```

   - **Linux inside WSL (Windows Subsystem for Linux):** Pass `--server.headless true` to prevent browser launch errors:

     ```bash
     uv run streamlit run app.py --server.headless true
     ```

## 🔄 6. Package Maintenance & Updates

Whenever you need to add, upgrade, or sync project packages:

```bash
# Add a new package
uv add <package_name>

# Upgrade all packages in uv.lock
uv lock --upgrade

# Or upgrade a single specific package (e.g., streamlit)
uv lock --upgrade-package streamlit

# Sync your local .venv with the updated uv.lock
uv sync
```

## ☁️ 7. Deployment to Streamlit Community Cloud

1. Push your repository to GitHub (ensure `app.py`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitattributes`, `.gitignore`, and your `data/` folder are committed).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select your GitHub repository, branch (`main`), and main file path (`app.py`).
4. Under **Advanced settings > Secrets**, paste your production API key using the exact same TOML format shown in Section 4.
5. Click **Deploy!** Streamlit Community Cloud natively detects `uv.lock` and builds the environment automatically.
