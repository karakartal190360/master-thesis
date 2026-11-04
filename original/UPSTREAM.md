# Official PVP source

- Repository: <https://github.com/JunXue-tech/PVP>
- Commit: [`1eed3f43309e6728583a5e43595e68121472b2ca`](https://github.com/JunXue-tech/PVP/commit/1eed3f43309e6728583a5e43595e68121472b2ca)
- Commit date: 2026-08-11; inspected on 2026-11-03.
- `PVP/` is a complete, unchanged export of the 23 files in this commit, including
  protocol CSVs and the architecture figure. It contains no nested Git repository.
  About 29 MB of source and metadata; no audio, weights, or experiment outputs.
- `manifest.json` records each upstream Git blob ID, SHA-256 hash, and byte size.
  All 23 file contents were verified against the pinned Git tree during setup.
- No `LICENSE` file exists in the inspected commit. This records source provenance;
  it does not assign a new license to the authors' code.

Keep `PVP/` unchanged. Put our code in `src/` and `scripts/` when implementation begins.
If a patch is needed, keep it outside the snapshot and record the reason and exact
behavioral difference in `docs/reproduction.md`. An upstream update needs a deliberate
commit change, a regenerated manifest, and a new paper/code comparison.

Do not run `install.sh` or import the experiment scripts as a setup check yet.
The installer is not a complete lockfile, and imports can load/download models.
Read `../docs/environment.md` first.
