# Practical Work: Personalized Speech Deepfake Detection

This project studies how to detect AI-generated speech using genuine recordings
of the same speaker. It follows [Xue et al.'s PVP method](Papers/profiling-the-voice.pdf).
The pipeline identifies phonemes, which are individual speech sounds, extracts
audio features, and builds a speaker profile. It compares new recordings with
that profile to calculate a detection score.

The `practical-work` branch contains only the practical work. Future master's
thesis work will continue in a separate `master-thesis` branch based on this work.

## Planned work

1. **Complete the reproducible pipeline.** Record settings, seeds, code versions,
   model and data versions, and results. Track experiments with Weights & Biases.
2. **Reproduce the original experiments.** Use Xue's ZH-Famous and Famous Figures
   datasets and supplied protocols. Compare our measured results with the paper.
3. **Evaluate additional datasets and benchmarks.** Choose suitable ones later.
   Keep these evaluations separate from the original reproduction.
4. **Write the report.** Describe the method, experiments, results, problems,
   and limitations.
5. **Add an interface if time allows.** After the pipeline works, support audio
   uploads and visual explanations. Graphs or animations could show phoneme scores
   and their influence on the final score.

## Current status

CUDA software checks and W&B setup pass. The pretrained models are downloaded.
Model loading checks and pipeline implementation remain pending. Dataset licenses
remain under review. No reproduction results exist yet.

See the [setup instructions](docs/environment.md),
[reproduction notes](docs/reproduction.md), and
[original repository](https://github.com/JunXue-tech/PVP).
The [main branch README](https://github.com/karakartal190360/master-thesis/blob/main/README.md)
keeps the detailed project and setup information.
