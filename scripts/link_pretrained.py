"""Create the runtime layout that upstream PVP expects, using symlinks to local files.

Upstream resolves model names relative to its working directory. Links let the
unchanged code find our pinned files, so it never downloads anything.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
UPSTREAM = ROOT / "original/PVP"


def link(target, link_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() and link_path.resolve() == target.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        raise SystemExit(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(target)


def link_pretrained(work_dir, models=MODELS):
    work_dir = Path(work_dir)
    # backbones/mms.py: torch.load("pretrained_models/mms_300m.ckpt")
    link(models / "mms-300m-anti-deepfake/mms_300m.ckpt",
         work_dir / "pretrained_models/mms_300m.ckpt")
    # 2_PVP.py: SpeechBrain treats source "speechbrain/..." as local only if that
    # directory exists; otherwise it always checks the Hub (allow_updates=True).
    ecapa = models / "spkrec-ecapa-voxceleb"
    link(ecapa, work_dir / "speechbrain/spkrec-ecapa-voxceleb")
    # SpeechBrain savedir; per-file links so its writes never reach models/.
    for path in sorted(ecapa.iterdir()):
        if path.is_file():
            link(path, work_dir / "pretrained_models/spkrec-ecapa-voxceleb" / path.name)
    # analyzer.py: from_pretrained("facebook/...") loads a local directory of that name first.
    link(models / "wav2vec2-xlsr-53-espeak-cv-ft",
         work_dir / "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    # analyzer.py: open("vocab.json"), upstream phoneme-category mapping.
    link(UPSTREAM / "vocab.json", work_dir / "vocab.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    args = parser.parse_args()
    link_pretrained(args.work_dir)
    print(f"Linked pretrained files into {args.work_dir}")


if __name__ == "__main__":
    main()
