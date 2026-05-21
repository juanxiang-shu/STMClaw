# Autonomous STM Control and Tip Conditioning Framework

A Python-based research framework for autonomous STM operation and tip conditioning, designed for deployment within an OpenClaw-compatible STM automation workflow.

---

## Overview

This repository implements a research-grade STM automation engine for high-resolution scanning, adaptive tip conditioning, and intelligent path planning.

The toolkit integrates:
- real-time STM control via Nanonis TCP interface
- image-based scan quality evaluation
- autonomous tip conditioning strategies
- customizable path planning and waypoint navigation
- optional molecule matching using Gemini / LLM-assisted inference

The core scan workflow is implemented in `Auto_scan.py`, and is intended to be launched through an OpenClaw STM orchestration layer or equivalent runtime wrapper.

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
git clone <repository-url> STMClaw
cd STMClaw
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

Example:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
$env:CROSSREF_EMAIL = "your.email@example.com"
```

---

## Usage

This repository is designed to run as the STM scan engine within an OpenClaw automation deployment.

In production, your OpenClaw wrapper should:

1. set the repository root as the working directory;
2. activate the Python environment containing the project dependencies;
3. export required runtime variables such as `GEMINI_API_KEY` and `CROSSREF_EMAIL`;
4. launch `Auto_scan.py` as the core scan process; and
5. manage logging, error recovery, and instrument session lifecycles.

For development and debugging, the underlying scan engine is exposed by `Auto_scan.py`.

A typical OpenClaw deployment flow is:

```powershell
cd "STMClaw"
.\.venv\Scripts\activate
$env:GEMINI_API_KEY = "your_api_key"
$env:CROSSREF_EMAIL = "your.email@example.com"
python Auto_scan.py
```

`Auto_scan.py` initializes the STM controller, configures the navigator, and enters the autonomous scanning loop.

For targeted agent testing and development, use the module entry points in `modules/` and `Evaluation/`.

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
