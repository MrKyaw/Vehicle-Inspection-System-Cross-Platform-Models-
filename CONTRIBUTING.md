# Contributing

Thanks for looking at the project. This repository is primarily a portfolio/research artifact, but the engineering layer is maintained as a normal Python package.

## Ground rules

1. **Preserve the research artifacts.** Do not delete or rewrite the notebooks, `src/lib.rs`, Flask/FastAPI samples, `index.html`, `resnet18.onnx`, images, or `License.text`. Changes there should be additive and clearly explained.
2. **Runtime package stays dependency-free.** `vehicle_inspection` may import only the Python standard library. Test tooling belongs in the `dev` extra.
3. **No network in tests.** Tests must not perform HTTP, DNS, model downloads, or package installation.
4. **Coverage stays at 100%** line and branch for `vehicle_inspection`. Do not add `# pragma: no cover` to reachable logic; remove dead code or test the branch.
5. **Determinism.** No random fixtures, no wall-clock assertions, no dependence on dict iteration order for decisions.

## Setup

```bash
git clone https://github.com/MrKyaw/Vehicle-Inspection-System-Cross-Platform-Models-.git
cd Vehicle-Inspection-System-Cross-Platform-Models-
python3 -m pip install -e ".[dev]"
```

Python 3.10, 3.11, and 3.12 are supported.

## Before opening a pull request

```bash
python3 -m pytest -q
python3 -m pytest --cov=vehicle_inspection --cov-branch --cov-report=term-missing
python3 -m vehicle_inspection --demo
python3 -m vehicle_inspection --help
```

All four must succeed, and the coverage `Missing` column must be empty.

## Style

- Type hints on public functions and dataclasses.
- Docstrings that explain *why* and state invariants, not just restate the signature.
- Frozen dataclasses (`slots=True`) for value objects.
- Domain errors derive from `VehicleInspectionError`; never let raw `KeyError`, `ValueError`, or `TypeError` escape a public API for malformed input.
- Keep functions small and side-effect free unless they are explicitly I/O adapters.
- Prefer explicit loops over clever comprehensions when the loop has meaningful branches; it keeps coverage honest and debugging simple.

## Tests

Add tests to `tests/test_vehicle_inspection.py`.

- Cover the happy path, the boundary, and at least one malformed input for every new public behavior.
- For threshold logic, test just below, exactly at, and just above the threshold.
- For NMS, test score ties, IoU exactly at the threshold, and empty input.
- For CLI changes, use `tmp_path`, `capsys`, and `monkeypatch`; assert exit codes and stderr.
- Use `pytest.approx` for floating-point values.

## Commit and PR notes

- Keep commits focused; describe the behavior change and the test evidence.
- State any change to the public API or report schema explicitly.
- Update `README.md`, `ARCHITECTURE.md`, `TESTING.md`, or `docs/DESIGN_DECISIONS.md` when behavior or rationale changes.
- Do not commit generated artifacts: `.coverage`, `htmlcov/`, `__pycache__/`, `.pytest_cache/`, build outputs.

## Reporting issues

Include:

- Python version and operating system.
- The command you ran.
- The input manifest/detections (redact anything sensitive).
- Expected vs actual verdict or error.
- Whether the issue is in the new package, the research notebooks, or a deployment sample.

## License

By contributing, you agree that your contribution is licensed under the repository's MIT license.
