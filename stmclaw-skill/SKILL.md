---
name: stmclaw
description: "Control STMClaw autonomous scanning tunneling microscope scans."
---

# STMClaw Skill

Control STMClaw for autonomous STM scanning, tip conditioning, and molecule assignment.

## Prerequisites

- Python environment at `I:\STMClaw\stmclaw-env`
- GEMINI_API_KEY environment variable set
- Nanonis TCP interface available

## Tools

- `start_scan` - Start an autonomous STM scan with optional custom navigation instruction

## Usage

### Start a default scan

Use stmclaw to start a scan

### Start with custom instruction

Use stmclaw to start a scan from the center in a spiral pattern

## Implementation

The skill delegates to STMClaw's openclaw_adapter.py via the scripts/start_scan.py script.
