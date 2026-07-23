# FreshGrad Colleague Contribution

These artifacts were received from a colleague and imported on 2026-07-23 for
team review:

- `freshgrad-v001.pptx`: a nine-slide problem, method, experiment, and roadmap
  deck.
- `microlearning-song.ipynb`: a microlearning comparison of backpropagation,
  Hebbian learning, and predictive coding.
- `research-song.ipynb`: static-MNIST, continual-learning, and
  inference-budget experiments comparing backpropagation and predictive coding.
- `predictive-coding-report-3-implementations.html`: a self-contained
  interactive report covering three predictive-coding implementations.

## Review Status

The files are contributed research artifacts, not canonical project evidence.
Their result statements and embedded notebook outputs have not been reproduced
or scientifically audited in this repository. Both notebooks arrived with
unexecuted code-cell metadata, so the embedded outputs are not traceable to an
execution recorded by Jupyter. Treat their claims as working material until
matched runs and source checks satisfy the project contract.

The imported `research-song.ipynb` has one portability cleanup: a
contributor-specific absolute MNIST path and its run note were replaced with
the repository-relative `./data` path. The imported HTML also had whitespace
removed from nine otherwise blank lines. The code still uses `download=False`,
so MNIST must already be present there. Although the research notebook's
docstring and final console message name `pc_cl_results.pkl`, the contributed
code does not call `pickle.dump` and therefore does not currently write that
artifact.

## Reproduction

From this directory, with Jupyter, PyTorch, torchvision, NumPy, and Matplotlib
installed:

```bash
python -m jupyter nbconvert --to notebook --execute \
  microlearning-song.ipynb --output microlearning-song.executed.ipynb

python -m jupyter nbconvert --to notebook --execute \
  research-song.ipynb --output research-song.executed.ipynb
```

These are the smallest execution commands, not a claim that the full runs have
passed locally. The notebooks may take substantial time, and the research
notebook expects MNIST under `./data`.

## Received-File Checksums

SHA-256 checksums before the portability cleanup:

```text
a2dc47e0e73ad3631564b3bab7b599778d160165393a22950a30f77e7b2662ff  FreshGrad V001.pptx
67a1c25d816e74525b3fc4a8ec99d922c351fec4adc8ddb3796fff45f79a9550  Microlearning-Song.ipynb
a444d507a2ecbd8b29235cb4b030888c66acf71ceb3828f951db0ca39dcd5e72  PredictiveCoding_Report_3_Implementations.html
b29e4badac7d892f6603c7dac60cb1c56d8813a1a740e1b111c94b22a1a3032b  Research-Song.ipynb
```
