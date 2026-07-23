# Deferred multi-host compute extension

**Status:** deferred planning note; not part of
`matched-lr-state-policy-v1`, not a registered protocol, and not authorization
to start or distribute study cells.

## v1 boundary

`matched-lr-state-policy-v1` runs sequentially on the stable local RTX 4090.
The RTX 3060, WD SSD, SSH alias, bundles, artifact transfer, cross-GPU
calibration, hardware assignment, seed migration, and gaming computer do not
apply to v1.

The completed `matched-lr-paper-v1` study also remains unchanged. Its protocol
required sequential CUDA execution on the RTX 4090. Nothing in this note
retroactively changes that source or result tree.

## Why retain this note

A later protocol version may need more throughput. The earlier dual-host
design identified useful requirements for such an extension, but those
requirements add enough operational and numerical uncertainty that they
should be evaluated only after the learning-first local study is understood.

The 24 validated `matched-lr-paper-v1` result files recorded PyTorch CUDA peaks
from 74.60 to 77.59 MiB, with a maximum of 81,355,264 bytes. These allocation
figures suggest that the current MLP is well below a 12 GiB capacity ceiling,
but they do not establish total process VRAM, deterministic equivalence, or
operational suitability on another GPU.

## Candidate multi-host design for a future protocol

If a new protocol authorizes the RTX 3060 and RTX 4090 together:

1. Treat each seed's eight conditions as an indivisible hardware block.
2. Assign seed blocks before inspecting outcomes, using timing only.
3. Counterbalance condition order within seed using the frozen Williams rows.
4. Keep source, submodule, dependency lock, dataset, stream, initialization,
   sample/update counts, and receipt identities exact across hosts.
5. Require two complete smoke passes on each GPU before assignment.
6. Establish deterministic repetition within each GPU. Treat cross-architecture
   equality as an empirical calibration question rather than assuming a
   tolerance after seeing results.
7. Keep runtime and peak memory stratified by GPU identity, or designate one
   GPU for the primary resource benchmark.
8. Write to separate host/attempt directories and curate only sealed,
   independently validated result/receipt pairs.
9. Freeze assignment, failure, recovery, transfer, and interruption policy
   before primary cells.

Bridge seed 42 could be duplicated across devices as an explicitly
non-confirmatory hardware check. It must remain outside held-out effect
summaries.

## SSH and bundle boundary for a future protocol

A future implementation may reuse the following safety ideas:

- resolve one explicit SSH alias and require a pinned host key;
- require `IdentitiesOnly yes`, `ForwardAgent no`, and strict host-key
  checking;
- use an unprivileged remote account and avoid recording effective secret
  values;
- create separate Git bundles for the main repository and predictive-coding
  submodule;
- verify commits, trees, the main-repository gitlink, clean status, source
  hashes, and bundle prerequisites on both hosts;
- recreate the environment from the frozen lock rather than transferring a
  virtual environment;
- transfer MNIST raw files only with exact SHA-256 verification;
- launch remote work durably and record the PID and sealed artifact paths;
- after connection loss, probe the PID and artifacts before considering a
  retry; never launch a blind duplicate;
- pull sealed artifacts only between cells and verify hashes before
  acceptance.

These are design candidates, not implemented v1 behavior.

## Storage and recovery boundary for a future protocol

If a dedicated SSD is used later, preflight should verify the exact mount,
filesystem, read/write and executable behavior, and free capacity without
changing mount, ownership, or permission policy. Bundles, controller state,
worker checkout, attempts, and curated archive should have separate,
non-overwriting paths.

A valid pair should never rerun. An interrupted attempt should be preserved,
reviewed, and followed by a new attempt directory. A result, invariant,
source, hardware, or transfer failure should stop new launches and produce a
hash-addressed human-review record.

## Intermittently used gaming computer

An interruptible gaming GPU is unsuitable for a required primary cell unless a
future protocol explicitly reserves it for the entire cell and records:

- no game, overlay, renderer, inference job, or graphics/compute workload;
- stable driver, operating-system, dependency, and power policy;
- pre-cell idle gates and during-run telemetry;
- sufficient uninterrupted time for execution, validation, and sealing;
- immediate invalidation or interruption handling if a foreign workload
  starts.

Checking only CUDA compute processes is insufficient because a game can use
graphics queues and VRAM. Without an exclusive reservation, use such a machine
only for unit tests, smoke/performance reconnaissance, disposable diagnostics,
or an explicitly non-primary replication.

## Registration trigger

Before any multi-host cell, create a new protocol version and obtain human
review of:

- the scientific reason to expand beyond the local RTX 4090;
- per-GPU commissioning evidence;
- cross-device numerical policy;
- timing-only seed assignment and run order;
- resource-reporting policy;
- SSH, bundle, dataset-transfer, storage, idle, interruption, and recovery
  procedures;
- the exact relationship of any hardware bridge to confirmatory seeds.

Until that checkpoint, every multi-host idea in this note remains deferred.
