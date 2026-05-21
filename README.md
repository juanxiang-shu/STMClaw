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

For production deployment, OpenClaw should:

1. set the repository root as the working directory;
2. activate the Python environment containing the project dependencies;
3. export required runtime variables such as `GEMINI_API_KEY` and `CROSSREF_EMAIL`;
4. invoke STMClaw through the OpenClaw runtime so that OpenClaw manages instrument sessions, execution lifecycle, logging, and clean shutdowns; and
5. handle any recovery or retry logic around STM execution.

This repository includes a minimal OpenClaw compatibility bridge:

- `openclaw_adapter.py` — a adapter module that exposes `run_stmclaw()` as an entrypoint
- `openclaw-config.yaml` — a sample OpenClaw deployment configuration

A typical OpenClaw deployment flow is:

```powershell
cd "STMClaw"
.\stmclaw-env\Scripts\activate
$env:GEMINI_API_KEY = "your_api_key"
$env:CROSSREF_EMAIL = "your.email@example.com"
# Start the OpenClaw orchestration runtime, which loads STMClaw as the scan engine
openclaw run --config openclaw-config.yaml
```

> Note: `openclaw_adapter.py` is a minimal compatibility stub. Full production integration may require additional hooks for session lifecycle, retries, structured logging, and graceful shutdown.

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
