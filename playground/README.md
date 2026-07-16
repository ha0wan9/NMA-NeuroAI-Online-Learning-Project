# Playground

## Predictive Coding

`predictive-coding/` is a Git submodule that points to the Bogacz Group's
PredictiveCoding repository:

- Source: <https://github.com/Bogacz-Group/PredictiveCoding>
- Upstream owner: Bogacz Group
- Pinned commit: `5bf803c3636a39928488941f05539c70ab4df0b1`
- Added: 2026-07-16

The upstream repository did not include a license file when this submodule was
added. The source code remains in the upstream repository and is not vendored
here. Consult the upstream maintainers before copying, modifying, or
redistributing it.

Clone this repository with its submodules:

```bash
git clone --recurse-submodules <repository-url>
```

For an existing clone, initialize the submodule with:

```bash
git submodule update --init --recursive
```

## Related Reading

- Gaspard Oliviers, ["Training brain-inspired predictive coding models in
  Python"](https://medium.com/@oliviers.gaspard/training-brain-inspired-predictive-coding-models-in-python-5a7011e2779d)
  (Medium, 18 February 2024). A practical introduction to training and using
  the predictive-coding library linked above.
- Yuhang Song, Beren Millidge, Tommaso Salvatori, et al., ["Inferring neural
  activity before plasticity as a foundation for learning beyond
  backpropagation"](https://doi.org/10.1038/s41593-023-01514-1), *Nature
  Neuroscience* 27, 348--358 (2024). The paper introduces prospective
  configuration and evaluates it in predictive-coding networks across several
  learning settings, including online and continual learning. The article is
  available under the Creative Commons Attribution 4.0 International License.
