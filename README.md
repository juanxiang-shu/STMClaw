# Autonomous STM Control and Tip Conditioning Framework

An integrated Python framework for autonomous scanning tunneling microscopy (STM) operation, tip conditioning, and molecule matching using Nanonis TCP control, image quality evaluation, and Gemini-assisted planning.

---

## Overview

This repository implements a research-grade STM automation system designed for high-resolution scanning, adaptive tip conditioning, and path planning.

The toolkit combines:
- real-time STM control via Nanonis TCP interface
- image-based scan quality evaluation
- autonomous tip conditioning strategies
- customizable path planning and waypoint navigation
- optional molecule matching using Gemini / LLM-driven inference

The entry point is `Auto_scan.py`, which orchestrates scan execution, evaluation, conditioning, and planning.

---

## Key Features

- **Autonomous scanning loop** with real-time STM instrument interaction
- **Tip conditioning agent** for adaptive repair and tip stabilization
- **Path planning agent** supporting custom navigator instructions
- **Image quality evaluation** using CV and learned heuristics
- **Molecule matching / assignment pipeline** for structure-surface inference
- **Modular architecture** with clear separation between conditioning, planning, evaluation, and assignment

---

## Repository Structure

- `Auto_scan.py` — main driver implementing the scan workflow
- `core.py` — Nanonis TCP control abstraction and STM instrument interface
- `modules/conditioning_agent.py` — conditioning agent utilities and tip-conditioning logic
- `modules/planning_agent.py` — planning agent utilities, path generation, and Gemini integration
- `modules/evaluation.py` — scan quality evaluation helpers and image processing
- `modules/assignment_agent.py` — molecule matching and assignment utilities
- `mol_segment/` — segmentation model utilities and image processing for molecular feature extraction
- `tasks/LineScanchecker.py` — line-scan analysis helper functions
- `Evaluation/` — detection and evaluation scripts supporting experimental analysis

---

## Installation

This project is developed for Python and is best run in an isolated environment.

1. Clone the repository:

```bash
git clone <repository-url>
cd "SPM-nanonis_TCP - Copy"
```

2. Create and activate a Python environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install the main dependencies used by the code manually:

```bash
pip install numpy scipy opencv-python matplotlib keyboard google-genai torch pillow httpx pubchempy
```

---

## Configuration

The framework expects several environment variables for external API access and runtime configuration.

Required environment variables:

- `GEMINI_API_KEY` — Gemini / Google GenAI API key for LLM-assisted path planning and evaluation
- `CROSSREF_EMAIL` — email address used by paper search / assignment utilities

Optional variables:

- `GEMINI_MODEL` — default Gemini model name (e.g. `gemini-2.5-flash`)
- `GEMINI_MODEL_NAVIGATOR` — Gemini model for navigator requests
- `GEMINI_MODEL_PATH` — Gemini model for path-related requests
- `GEMINI_MODEL_CODEGEN` — Gemini model for code generation tasks
- `GEMINI_TIMEOUT` — request timeout in seconds
- `GEMINI_RETRIES` — number of retry attempts when Gemini calls fail
- `GEMINI_RETRY_BACKOFF` — backoff delay between Gemini retries
- `MOLECULE_MATCH_ENABLED` — enable or disable molecule matching (`1` / `0`)
- `MOLECULE_MATCH_INTERVAL` — evaluation interval for molecule matching

Example:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
$env:CROSSREF_EMAIL = "your.email@example.com"
```

---

## Usage

Run the main scan workflow with:

```bash
python Auto_scan.py
```

`Auto_scan.py` initializes the scan controller, configures the navigator, and enters the autonomous STM scanning loop.

For targeted agent testing and development, use the corresponding module entry points in `modules/` and `Evaluation/`.

---

## Notes for GitHub Publication

- Sensitive credentials are no longer stored in source code.
- Local workspace artifacts such as `.idea/`, `.vscode/`, virtual environments, logs, and temporary caches are excluded from version control.
- The code is organized to keep scientific logic separate from instrument control and model utilities.

---

## Contribution

This repository is intended as a research tool and can be extended with:

- additional STM motion strategies
- improved tip conditioning heuristics
- robust Gemini prompt engineering for path planning
- integration with new image segmentation models
- experiment logging and analysis dashboards

If you contribute, please follow clean Python packaging practices and avoid committing API keys or local configuration.

---

## License

Add an appropriate license file to this repository before publishing. Recommended options include `MIT`, `Apache-2.0`, or another license consistent with your institution's policy.
