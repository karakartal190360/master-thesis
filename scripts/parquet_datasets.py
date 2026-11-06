"""Column names of the Hugging Face Parquet releases and a row reader shared by dataset scripts."""

import pyarrow.parquet as pq

# Common name -> release column. None: column does not exist in this release.
# speaker_splits: shards named "<Speaker>-000NN-of-000MM".
# strip_source_suffix: protocol keys may add "_<Source>" to the release path (see docs/reproduction.md).
DATASETS = {
    # PVP-ADD/ZH-Famous: one "train" split.
    "zh-famous": {"audio": "file_name", "label": "label", "speaker": "person",
                  "source": "method", "duration": "duration_sec", "name": None,
                  "speaker_splits": False, "strip_source_suffix": False},
    # issf/famousfigures: one split per speaker.
    "famous-figures": {"audio": "Audio", "label": "Label", "speaker": "Speaker",
                       "source": "Source", "duration": None, "name": "AudioName",
                       "speaker_splits": True, "strip_source_suffix": True},
}


def read_rows(shard, dataset, audio=False):
    """Yield one dict per row: file, label, speaker, source, duration, name, and bytes if audio."""
    columns = DATASETS[dataset]
    wanted = [columns["audio"] if audio else columns["audio"] + ".path"]
    wanted += [columns[k] for k in ("label", "speaker", "source", "duration", "name") if columns[k]]
    # Per row group: pq.read_table loads the whole audio struct (13 GB RAM for one ZH-Famous shard).
    parquet = pq.ParquetFile(shard)
    row = 0
    for group in range(parquet.num_row_groups):
        for r in parquet.read_row_group(group, columns=wanted).to_pylist():
            out = {"shard": shard.name, "row": row, "file": r[columns["audio"]]["path"]}
            for key in ("label", "speaker", "source", "duration", "name"):
                out[key] = r[columns[key]] if columns[key] else None
            if audio:
                out["bytes"] = r[columns["audio"]]["bytes"]
            yield out
            row += 1
