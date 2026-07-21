# H1.E3 Permuted MNIST manifest

- Slug: `H1E3-permuted-mnist`
- Intervention: PC relaxation steps `1/5/10/20`; all other fields frozen.
- Control: fresh BP, same paired seeds, initialization, and domain-incremental streams.
- Command: `bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh`
- Expected artifact: Permuted-MNIST section of `runs/relaxation-sweep-v1.json`.
- Protocol: study `index.md` and `02-design.md`.
