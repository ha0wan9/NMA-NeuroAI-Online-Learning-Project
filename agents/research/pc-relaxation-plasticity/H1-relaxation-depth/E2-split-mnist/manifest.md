# H1.E2 SplitMNIST manifest

- Slug: `H1E2-split-mnist`
- Intervention: PC relaxation steps `1/5/10/20`; all other fields frozen.
- Control: fresh BP, same paired seeds, initialization, and class-incremental streams.
- Command: `bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh`
- Expected artifact: SplitMNIST section of `runs/relaxation-sweep-v1.json`.
- Protocol: study `index.md` and `02-design.md`.
