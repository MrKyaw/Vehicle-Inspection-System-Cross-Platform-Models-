# Testing

## Policy

The new `vehicle_inspection` package is held to **100% line and branch coverage**, enforced twice:

- `pyproject.toml` → `[tool.coverage.report] fail_under = 100`
- CI runs the coverage command for every supported Python version

Coverage is a floor, not the goal. The suite is written around the behaviors that can hurt an inspection system: malformed manifests, inverted boxes, NMS ties, threshold boundaries, calibration that changes a verdict, unknown classes, and CLI failure paths.

## Setup

```bash
python3 -m pip install -e ".[dev]"
```

`pytest` and `pytest-cov` are the only development dependencies. The runtime package itself has **zero** dependencies.

## Commands

```bash
# Fast suite
python3 -m pytest -q

# Required coverage gate
python3 -m pytest --cov=vehicle_inspection --cov-branch --cov-report=term-missing

# Optional HTML review
python3 -m pytest --cov=vehicle_inspection --cov-branch --cov-report=html
open htmlcov/index.html
```

Or:

```bash
make test
make coverage
```

## Verified results

Local run on Python 3.12.8, pytest 9.1.1, coverage 7.16.2:

```text
372 passed
```

```text
Name                                Stmts   Miss Branch BrPart  Cover   Missing
vehicle_inspection/__init__.py         10      0      0      0   100%
vehicle_inspection/__main__.py          4      0      2      0   100%
vehicle_inspection/calibration.py     108      0     60      0   100%
vehicle_inspection/cli.py             136      0     46      0   100%
vehicle_inspection/demo.py             16      0      0      0   100%
vehicle_inspection/errors.py            7      0      0      0   100%
vehicle_inspection/geometry.py        103      0     40      0   100%
vehicle_inspection/manifest.py        215      0     88      0   100%
vehicle_inspection/policy.py          192      0    102      0   100%
vehicle_inspection/report.py           64      0     34      0   100%
TOTAL                                 855      0    372      0   100%
```

The `Missing` column is empty for every module. No `# pragma: no cover` is used to hide runtime logic.

## Test strategy

### Geometry (`geometry.py`)

- Box property math and center/area.
- Rejection of NaN/inf coordinates and inverted edges.
- Clamping inside, across all edges, and fully outside an image.
- IoU for identical, disjoint, touching, partial, and degenerate boxes.
- NMS: empty input, length mismatch, invalid thresholds, non-finite scores, score-floor skipping, strict IoU boundary, duplicate ties, equal scores without overlap, and descending output order.
- Letterbox for landscape, portrait, exact-size, and 1-pixel extreme aspect ratios.
- Inverse mapping round-trip and clamping of out-of-canvas boxes.

### Manifest (`manifest.py`)

- Valid full and minimal manifests, plus defaults.
- Unknown keys at every level.
- Every required field missing individually.
- Wrong JSON types, empty/duplicate labels, unsupported platforms, duplicate platforms.
- Invalid resize, mean, std, padding, color order, and layout.
- Bad thresholds and the `review < fail` invariant.
- Calibration labels not present in `labels`; non-positive/non-finite temperatures.
- Non-string metadata.
- Missing file, directory path, and invalid JSON, all normalized to `ManifestError`.

### Calibration (`calibration.py`)

- Stable softmax, including extreme logits.
- Temperature scaling at `T < 1`, `T = 1`, and `T > 1`; input renormalization; all-zero inputs.
- ECE: perfect calibration, biased confidence, empty bins, confidence exactly `1.0`, invalid bins/samples.
- Bucket boundaries at `0.5`, `0.75`, and `0.9`.
- Per-class calibrator: present/absent labels, stable sigmoid branches, `0`/`1` preservation, invalid labels/temperatures.

### Policy (`policy.py`)

- Detection parsing and every malformed field.
- Class rule validation and ordering.
- All nine zone branches (including exact 1/3 boundaries).
- Verdict paths: no findings, ignored-only, cosmetic pass, review, severe fail, aggregate-risk fail.
- Calibration changing a verdict in both directions (including below the class floor).
- Class rule severity overriding a detection's claimed severity.
- Deterministic ordering with equal scores and equal labels.
- Unknown classes fail closed; `max_findings` guard; invalid image dimensions.
- Serialization of findings and ignored detections.

### Report (`report.py`)

- Reports with and without manifests.
- `generated_at`, source, suppression count, and empty findings.
- Invalid result type and suppression counts.
- Stable sorted JSON and custom indentation.
- Markdown with all sections, minimal reports, escaping of pipes/newlines, and every missing required key.

### CLI (`cli.py`)

- Bare-list and object detection files, with and without image dimensions.
- Every malformed file shape, missing file, invalid JSON, and invalid image flag.
- Manifest application, label mismatch, score-floor filtering, and absent calibration.
- Markdown and JSON output; `--output` success and directory failure.
- Mutually exclusive demo/file handling and missing input.
- Domain errors return `2` with stderr; success returns `0`.
- `--help`, `--version`, `runpy` module execution, and normal module import.

## Determinism and isolation

- **No network.** No test performs HTTP, DNS, or package installation.
- **No optional dependencies.** Tests import only `vehicle_inspection` and the standard library plus pytest.
- **No wall-clock assertions.** The library accepts `generated_at`; the CLI provides a real UTC timestamp, but tests assert presence/shape, not a fixed time.
- **No random data.** All boxes, scores, and fixtures are literal.
- **Temporary files only.** CLI tests use pytest's `tmp_path`.
- **No global state.** Dataclasses are frozen and tests do not mutate module constants.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests:

| Job | Python | Command |
| --- | --- | --- |
| test | 3.10 | `pytest -q` + coverage gate |
| test | 3.11 | `pytest -q` + coverage gate |
| test | 3.12 | `pytest -q` + coverage gate |

Every matrix entry installs `.[dev]`, runs the plain suite, then runs the branch coverage command with `fail_under = 100`.

## Adding tests

1. Find the branch in the module and write the smallest input that takes it.
2. Prefer a boundary test over a broad integration test.
3. Keep assertions on observable behavior, not private helpers.
4. If a new branch is unreachable by design, remove the dead code rather than excluding it.
5. Run the coverage command and confirm `Missing` is empty.
