# STMClaw — Autonomous STM Control Framework

STMClaw is a Python-based framework for autonomous STM operation and tip conditioning, designed for deployment within an OpenClaw-compatible STM automation workflow.

## Overview

This repository implements an STM automation workflow for high-resolution scanning, adaptive tip conditioning, and intelligent path planning.

The toolkit integrates:
- real-time STM control via Nanonis TCP interface
- image-based scan quality evaluation
- autonomous tip conditioning strategies
- customizable path planning
- molecule assignment using LLM

The core scan workflow is implemented in `Auto_scan.py`, and is intended to be launched through an OpenClaw STM orchestration layer or equivalent runtime wrapper.

## Repository Structure

- `Auto_scan.py` — main driver implementing the scan workflow
- `core.py` — Nanonis TCP control abstraction and STM instrument interface
- `modules/conditioning_agent.py` — conditioning agent and tip-conditioning strategy
- `modules/planning_agent.py` — planning agent, path generation, and LLM integration
- `modules/evaluation.py` — scan quality evaluation helpers and image processing
- `modules/assignment_agent.py` — molecule assignment
- `openclaw_adapter.py` — minimal OpenClaw-compatible bridge for starting STMClaw
- `openclaw-config.yaml` — sample OpenClaw deployment configuration

## Installation

This project is developed for Python and is best run in an isolated environment.

1. Clone the repository:

```bash
git clone https://github.com/juanxiang-shu/STMClaw.git
cd STMClaw
```

2. Create and activate a Python environment:

```bash
python -m venv stmclaw-env
.\stmclaw-env\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Configuration

The framework expects several environment variables for external API access and runtime configuration.

Required environment variables:

- `GEMINI_API_KEY` — Google API key for LLM-assisted path planning and evaluation
- `CROSSREF_EMAIL` — email address used by paper search

Example:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
$env:CROSSREF_EMAIL = "your.email@example.com"
```

Optional runtime environment variables:

- `STMCLAW_NAVIGATION_INSTRUCTION` — natural-language instruction for the navigation planner, used by the OpenClaw adapter.

## Usage

This repository is intended to be consumed as the STM scan engine inside an OpenClaw-compatible orchestration layer.

### OpenClaw deployment

STMClaw is now packaged as an OpenClaw skill. The root [skill.yaml](skill.yaml) declares the skill metadata and the `start_scan` tool, and the packaged runtime assets are stored under [stmclaw-skill](stmclaw-skill).

For production deployment, OpenClaw should:

1. load the installed STMClaw skill (or register the repository root so the skill manifest is discoverable);
2. activate the Python environment containing the project dependencies, typically `stmclaw-env`;
3. ensure required runtime variables are available, either through the environment or via [stmclaw-skill/config.env](stmclaw-skill/config.env);
4. invoke the skill’s `start_scan` tool so OpenClaw manages the session lifecycle, logging, and shutdown; and
5. pass the navigation instruction through the skill arguments when a custom scan route is required.

The packaged skill entrypoints are:

- [skill.yaml](skill.yaml) — skill manifest and declared tool(s)
- [stmclaw-skill/SKILL.md](stmclaw-skill/SKILL.md) — skill usage notes
- [stmclaw-skill/scripts/start_scan.py](stmclaw-skill/scripts/start_scan.py) — wrapper that loads [stmclaw-skill/config.env](stmclaw-skill/config.env) and launches the adapter
- [openclaw_adapter.py](openclaw_adapter.py) — runtime adapter that executes the STMClaw workflow

A typical OpenClaw deployment flow is:

```powershell
# 1. Install the STMClaw skill (one-time setup)
openclaw skills install ".\STMClaw\stmclaw-skill" --as stmclaw

# 2. Activate the Python environment
cd ".\STMClaw"
.\stmclaw-env\Scripts\activate

# 3. Ensure GEMINI_API_KEY and CROSSREF_EMAIL are available in the environment

# 4. Start the OpenClaw gateway and invoke the STMClaw skill
openclaw gateway start
# Then use the skill via OpenClaw's tool system, e.g., call the stmclaw/start_scan tool
```

If you want to validate the packaged wrapper directly before wiring it into OpenClaw, run:

```powershell
cd ".\STMClaw"
.\stmclaw-env\Scripts\activate
python .\stmclaw-skill\scripts\start_scan.py --instruction "scan the center region in a spiral pattern"
```

> Note: the supported deployment path is now the packaged skill under [stmclaw-skill](stmclaw-skill). The adapter remains the runtime bridge that executes the actual scan logic.

### Local development

For local debugging and development, the scan workflow is still directly executable via `python Auto_scan.py`.

### Targeted testing

For module-level testing, use the entry points in `modules/` and `Evaluation/`.

## Contribution

This repository is intended as a research tool and can be extended with:

- additional STM motion strategies
- improved tip conditioning heuristics
- robust Gemini prompt engineering for path planning
- integration with new image segmentation models
- experiment logging and analysis dashboards

If you contribute, please follow clean Python packaging practices and avoid committing API keys or local configuration.

## License

Add an appropriate license file to this repository before publishing. Recommended options include `MIT`, `Apache-2.0`, or another license consistent with your institution's policy.
