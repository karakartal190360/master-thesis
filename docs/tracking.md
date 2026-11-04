# Experiment tracking

The project uses the user's W&B account with an enabled academic subscription. Configuration is in
[tracking.json](../configs/tracking.json): entity `umuttanriverdi`, project
`personalized-speech-deepfake-detection`. Keep credentials outside the repository.
The [project](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection)
is private.

Account verified through the authenticated W&B API on 2026-11-04: username
`umuttanriverdi03`, team `umuttanriverdi`, organization `umuttanriverdi-org`.
The subscription reports `status=ENABLED`, `subscriptionType=ACADEMIC_TRIAL`,
`expiresAt=null`, and plan display name `Enterprise`. The user confirmed that the
website describes the academic plan as free forever. Treat the plan as permanent
based on that confirmation; the API label alone does not establish its duration.
Exact API evidence is saved in
`results/environment/tracking-account.json` on the server and under
`results/environment/homeserver/` on the Mac. The technical tracking setup is complete;
full scientific experiment integration remains pending.

## Software setup

W&B 0.26.1 is pinned in [tracking.in](../requirements/tracking.in). It is the newest
inspected release supporting Python 3.9; W&B 0.27.0 and later inspected releases
require Python 3.10 or newer. The [tracking lock](../requirements/tracking-linux-py39.lock)
was resolved against the verified PVP lock. Installation added 12 packages while
preserving all 95 existing package versions; the package consistency and CUDA checks
passed afterwards.

The Linux setup script installs both dependency sets. To add tracking to an existing
WSL project environment without reinstalling the research stack:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv pip install --python .venv/bin/python --require-hashes -r requirements/tracking-linux-py39.lock
uv pip check --python .venv/bin/python
```

The server already has a working W&B credential. On a new machine, authenticate
interactively with `.venv/bin/wandb login`. Do not put the API key in JSON configs,
shell scripts, Git, or chat messages.

## Verify logging

From the WSL project directory, first create a fresh environment report:

```bash
.venv/bin/python scripts/check_environment.py --pvp --cuda --output results/environment/linux.json
.venv/bin/python scripts/check_tracking.py --offline
.venv/bin/python scripts/check_tracking.py
```

[check_tracking.py](../scripts/check_tracking.py) creates a small run with
`purpose=tracking_setup` and `scientific_result=false`. It logs the number of passed
environment checks and CUDA availability. It does not run PVP or invent detection
metrics. The offline mode makes no cloud run; its files remain local unless explicitly
synced. Online logging requires the configured W&B project and account access.

Each invocation has a separate directory under `results/tracking/`, containing:

- `run.json`: run ID, mode, URL, and prepared/completed/failed status.
- `artifacts/config.resolved.json`: exact tracking destination settings.
- `artifacts/environment.json` and `installed-packages.json`: runtime evidence.
- `artifacts/provenance.json`: executing Git commit/branch/remote, dirty status,
  complete porcelain status including untracked files, upstream commit, and source hashes.
- `artifacts/git-diff.patch`: staged and unstaged tracked changes, preserving whitespace.
- `artifacts/source/`: the exact setup scripts, configs, locks, and upstream manifest
  used by the check. Other untracked project files are listed, not automatically uploaded.
- `wandb/`: the SDK's local files and logs.

Online runs upload the artifact directory as a versioned `reproducibility` artifact
and wait for upload confirmation. Local records are created before contacting W&B.
The script disables automatic code scanning, console capture, and system-statistics
sampling; it logs the explicitly selected evidence. Generated files are ignored by Git.

Verified on 2026-11-04: offline logging passed, and online run
[`098lnlvu`](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection/runs/098lnlvu)
finished successfully. Its artifact `tracking-setup-098lnlvu:v0` is committed in
W&B; all 12 evidence files matched their local SHA-256 hashes after downloading
them back. Verification is saved in `results/environment/tracking-verification.json`
on the server, with a copy under `results/environment/homeserver/` on the Mac.
The setup run records the dirty checkout explicitly and has `scientific_result=false`.

## Rules for scientific experiments

The setup check may run from a dirty checkout because it is explicitly non-scientific.
The future experiment runner must enforce these accepted requirements:

- Official results require a clean, committed executing checkout. Capture the server's
  revision and status; the Mac and server copies do not synchronize automatically.
- Save a fully resolved experiment configuration, its schema/version/hash, the command,
  seeds, deterministic-operation settings, and package/system versions for every run.
- Use Xue's official datasets and enrollment/test protocols for the initial baseline.
  Pin releases and hash data, splits, pretrained weights, and any documented conversion.
- Retain all fitted profiles, per-trial scores, metrics/units/definitions, logs, and
  failure/fallback accounting. Never silently drop failed trials or tune on test data.
- Each rerun gets a new ID. Verify replay against the original scores/metrics using
  declared comparison criteria. Keep run artifacts immutable and backed up.
- Save the complete run record locally as well as in W&B. Keep bulk datasets, shared
  weights, and large feature caches on the server; avoid repeated cloud copies.

The intended execution order is configuration/provenance validation, deterministic
initialization, CUDA feature extraction, CPU GMM profile fitting, test scoring,
evaluation/artifact finalization, then replay verification. These are requirements
for the upcoming experiment pipeline, not functionality provided by this setup check.
