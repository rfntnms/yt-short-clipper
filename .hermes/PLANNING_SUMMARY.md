# YT-Short-Clipper v2 Migration — Planning Summary

All issues for Milestone 0 to Milestone 8 have been successfully created in Linear.
Total: 26 issues (RFN-15 to RFN-40).

## Milestones Created
- Milestone 0: Migration Preparation
- Milestone 1: Foundation
- Milestone 2: Core Pipeline MVP
- Milestone 3: Testing & Reliability
- Milestone 4: Gradio UI MVP
- Milestone 5: Batch & Scheduling
- Milestone 6: Portrait Smart Crop & Split Mode
- Milestone 7: Docker Deployment
- Milestone 8: Integration Validation

## Current Blockers / MCP Issues
- The Linear MCP server experienced repeated timeouts and disconnects during dependency linking.
- All issues are present, but their `blockedBy` relations are incomplete.
- To see the full dependency mapping, refer to `LINEAR_WORKFLOW.md` and the initial text blocks in `.hermes/LINEAR_PENDING_ISSUES.md`.

## Execution Order (First 5 Issues to Work On)

1. **RFN-15: Repo Audit & Safety Preparation** (Phase: Preparation)
   - Why: Establishes safe boundaries and baseline.
   - Depends on: None.
   - Criteria: `MIGRATION_MAP.md` created, v1 isolated.
   - Test: `pytest` or `python -m compileall`.

2. **RFN-17: Structured Logging & System Health Checks** (Phase: Foundation)
   - Why: Needs to be in place before any other code runs.
   - Depends on: RFN-15.
   - Criteria: `utils/logger.py` with `app.log` rolling, GPU check.
   - Test: Run logger script, check terminal output & file.

3. **RFN-18: Configuration Management System** (Phase: Foundation)
   - Why: Everything else consumes config.
   - Depends on: RFN-15, RFN-17.
   - Criteria: Loads `config.json`, hides API keys on print.
   - Test: `test_config_manager_load_save`.

4. **RFN-19: OpenAI-Compatible AI Client Factory** (Phase: Foundation)
   - Why: Required for transcriber and highlight detector.
   - Depends on: RFN-18.
   - Criteria: Instantiates single `openai.OpenAI`.
   - Test: Ensure it passes `base_url`.

5. **RFN-20: Baseline requirements.txt for v2** (Phase: Foundation)
   - Why: Fixes dev environment so pipeline can be built.
   - Depends on: RFN-15.
   - Criteria: `pip install -r requirements.txt` cleanly.
   - Test: Clean virtualenv install.

## Next Development Prompt
"Mulai kerjakan RFN-15 (Repo Audit) — tolong baca CONTEXT.md dan bikin MIGRATION_MAP.md sesuai aturan."