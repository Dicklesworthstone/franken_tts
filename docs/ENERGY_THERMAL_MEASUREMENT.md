# OQ-17: Energy and thermal measurement

This is the measurement contract for the throughput-profile energy row and the
Qwen/Pocket bakeoff. Its consumer is the scorecard's `joules/generated-minute`
gate in §15; its observed defect class is heat-soak and background drift, which
can change an unchanged benchmark by 30% or more. The contract is retained
because the harness writes a `valid_for_claim` field that the scorecard must
reject when false. It is not a product-progress metric.

## Claimed quantity and scope

For a fixed, correctly generated corpus with known decoded duration `A` seconds,
the harness reports:

`J_per_generated_minute = 60 * measured_domain_joules / A`.

The default domain is deliberately local to a SKU:

- macOS / Apple Silicon: the `powermetrics` estimated SoC sum of CPU, GPU, and
  ANE sample energy. It excludes display, storage, PSU loss, and any rail that
  `powermetrics` does not expose.
- Linux / Intel or AMD: the sum of *top-level package* RAPL powercap zones,
  `energy_uj`; children are never added because they overlap their parent.

This is not a process-energy attribution and not a wall-plug measurement.
`powermetrics` itself warns that its average-power estimates are unsuitable for
cross-device comparisons. Therefore this row may compare variants only on the
same SKU, OS build, power mode, and measurement domain. A cross-SKU bakeoff
must separately use one calibrated external wall meter for every candidate;
neither this harness nor a RAPL row can establish that claim.

## Preconditions

- Build and fixture identity, CPU feature string, thread count, execution
  profile, packet size, model/artifact hashes, exact command and environment are
  captured alongside the JSONL result. The command itself may contain sensitive
  input and is intentionally not printed by the harness.
- The corpus, seed, voice pack, and generated duration are identical for every
  arm. `--generated-seconds` is decoded audio duration, never input-text time.
- Measure after a warmup, with network synchronization, indexing, downloads,
  and other heavy applications stopped. Do not change governors, fan curves, or
  power limits for this procedure. Any such OS tuning needs a separately
  authorized, recorded experiment.
- Each observation must contain at least three 1-second sensor samples. Use a
  corpus long enough to satisfy that lower bound; the sustained row is 30
  minutes (or the fixed-long corpus approved by the scorecard).

## Platform collection

`benches/energy_thermal_bench.sh` stores raw samples forever under its requested
artifact directory, then appends an observation to `energy-thermal.jsonl`.
There is no cleanup step because raw evidence is part of the claim.

### macOS / Apple Silicon

The harness invokes:

```bash
sudo -n powermetrics --format plist --samplers cpu_power,thermal \
  --sample-rate 1000 --sample-count -1 --output-file RAW
```

It sums `processor.cpu_energy`, `processor.gpu_energy`, and
`processor.ane_energy` from NUL-separated plist samples. On the verified
Darwin 25 `powermetrics` plist surface these fields are millijoules (the
observed energy equals `power * elapsed`); the harness records the raw plist so
that an OS change can be revalidated before new claims. Any thermal pressure
other than `Nominal` makes the observation `valid_for_claim:false`.

### Linux / Intel and AMD

The harness accepts only top-level `/sys/class/powercap/*-rapl:[0-9]*` zones
that expose both `energy_uj` and `max_energy_range_uj`. It takes start/end
snapshots and corrects a single wrap using `max_energy_range_uj`. It refuses
the older AMD HWMON-only path because it does not expose an unambiguous counter
range for a long measurement. Enable a current kernel's AMD RAPL/powercap
driver rather than substituting a guessed wattage.

The Linux powercap interface defines `energy_uj` as a current energy counter
and `max_energy_range_uj` as its range; AMD's documented energy counters are
also RAPL MSRs. Record which domains were present in `rapl-deltas.tsv`. Linux
thermal-zone data is platform-specific, so this initial harness marks it
`unavailable` rather than inventing a temperature threshold; the sustained
receipt must attach the raw thermal-zone/frequency trace for the particular
host before publication.

## Thermal-paired A/B protocol and gates

1. Use fixed corpus and warm artifact cache. Warm each arm once; warmups are
   not observations.
2. For each of at least five pairs, execute the arms in **ABBA** order inside a
   single thermal window. Pre-register a randomized first order (`ABBA` or
   `BAAB`) and alternate thereafter.
   Sequential full-pass A then full-pass B is inadmissible.
3. Run the harness once per observation, retaining `RAW` and the JSONL line.
   If any macOS result has thermal pressure, retain it but reject it for a
   performance claim; rerun after the system returns to nominal pressure.
4. A scorecard row is publishable only when each arm has at least ten valid
   observations, coefficient of variation <= 5% for both joules/generated-minute
   and RTF, and the paired 95% confidence interval is retained. A value above
   5% is **NO VERDICT**, not a slower/faster result. A 30-minute sustained row
   additionally reports first-five-minute vs final-five-minute RTF and energy
   rates; any thermal/throttling event invalidates its headline row.

The harness's result is an observation, not a passing scorecard. It proves only
that the named hardware domain was sampled over this command. It does not prove
audio quality, cross-device efficiency, process-only energy, or a bakeoff win.

## Invocation

For a single observation, call the collector with that slot's arm label and the
same decoded duration:

```bash
benches/energy_thermal_bench.sh \
  --label qwen-a1 --generated-seconds 600 --output-dir tests/artifacts/energy \
  --command 'target/release/ftts say --profile throughput --fixture long-corpus'
```

The harness requires a noninteractive `sudo` credential on macOS and fails
closed when it cannot obtain samples. The next scorecard implementation must
consume schema version 1, reject `valid_for_claim:false`, calculate CV and the
paired confidence interval, and bind the workload fingerprint to every row.

For the mandatory paired schedule, use the ABBA driver. It alternates ABBA and
BAAB orders, takes five pairs (ten observations per arm), writes
`abba-summary.tsv`, and exits 65 rather than publish a verdict when either
arm's CV exceeds 5%:

```bash
benches/energy_thermal_abba.sh \
  --a-label qwen --a-command 'target/release/ftts say --profile throughput --fixture long-corpus' \
  --b-label pocket --b-command 'target/release/ftts say --profile throughput --fixture long-corpus' \
  --generated-seconds 600 --first-order baab --output-dir tests/artifacts/energy
```

## Collector trial — not a scorecard result

On the Darwin 25 / Mac16,11 development host, four consecutive 10-second
fixed-CPU collector observations after a 30-second preheat were all `Nominal`,
but varied from 18.5901 to 33.8759 J/generated-minute (normalised only for the
trial), CV 22.38%. The raw plist samples were retained under
`/tmp/frankentts-oq17-preheated/`. This **fails** the 5% gate and is not a TTS
or energy-efficiency claim. It demonstrates that the collector produces the
schema and, more importantly, that this shared host is not an admissible
publication environment; the required next trial is a clean-host ABBA run with
the real fixed TTS corpus.

## Sources

- [Linux kernel powercap documentation](https://docs.kernel.org/power/powercap/powercap.html)
  defines power zones and their `energy_uj`/`max_energy_range_uj` attributes.
- [Linux kernel AMD energy-driver documentation](https://www.kernel.org/doc/html/latest/hwmon/amd_energy.html)
  documents the AMD RAPL energy-counter source and scaling.
