# H1.E1 paper smoke

Command: `bash playground/step2-cifar10/remote-run.sh`.

Prepared gate: one paper seed, one pass, 40 train examples/class and 20 test
examples/class; preserve the artifact and use synchronized runtime to project
the full H1.E2 cost. The original 16-example/class smoke is retained as an
invalid run because its 160-example dataset yielded no full batch under the
paper's `batch_size=200, drop_last=True` semantics.
