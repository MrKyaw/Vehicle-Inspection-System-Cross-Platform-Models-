# Design decisions

Short architecture decision records for the engineering layer added to the original research prototype. Each entry states the context, the decision, and the consequence.

---

## ADR-001: Standard library only for the decision package

**Context.** The repository already demonstrates PyTorch, ONNX Runtime, Pillow, Flask, FastAPI, and Rust/WASM. Those dependencies are appropriate for inference and serving, but they make the decision logic harder to run in CI, containers, and restricted review environments.

**Decision.** `vehicle_inspection` depends only on Python 3.10+ and its standard library. No NumPy, PIL, OpenCV, torch, or ONNX Runtime.

**Consequence.** Geometry, calibration, manifest validation, policy, and reporting run anywhere Python runs. Image decoding and tensor inference remain deployment concerns outside the tested package. A future service can import the package without pulling a GPU stack.

---

## ADR-002: Strict manifest validation

**Context.** Pre-processing drift is a common production failure: mean/std swapped, color order changed, input shape altered, threshold silently defaulted. A tolerant parser hides the bug until predictions are wrong.

**Decision.** Reject unknown keys, wrong types, duplicate labels, unsupported platforms, non-positive dimensions, invalid color/layout values, calibration entries for undeclared labels, and `review >= fail`.

**Consequence.** A malformed manifest fails at startup with a precise message. Adding a field is a deliberate schema change, not an accidental one.

---

## ADR-003: Deterministic NMS with explicit tie-breaking

**Context.** Detector post-processing often relies on library behavior for equal scores. That makes near-duplicate suppression dependent on sort implementation details.

**Decision.** Sort candidates by `(-score, index)` and suppress when IoU is **strictly greater** than the threshold. Return kept indices in descending score order.

**Consequence.** Equal-score inputs produce reproducible output. The strict comparison means a pair exactly at the threshold is kept, which is easy to test and document.

---

## ADR-004: Letterbox math in the library, pixels elsewhere

**Context.** The original Rust/WASM preprocessor resizes to 224×224; the browser demo creates a tensor from it. Detector boxes must map back to source pixels identically across runtimes.

**Decision.** Model the integer-resize rounding, padding offsets, scale, and inverse mapping as pure functions. Do not decode images.

**Consequence.** The Python layer and the Rust/JS path can be checked against the same formulas without sharing an image library. Extreme aspect ratios clamp to at least one pixel, avoiding zero-sized tensors.

---

## ADR-005: Calibration is monotonic and separate from ranking

**Context.** Raw detector confidence is often over-confident. Temperature scaling is a simple, interpretable correction that does not require a calibration framework.

**Decision.** Provide per-class sigmoid temperature scaling for detector scores, full-vector temperature scaling, ECE, and buckets. Apply calibration before policy thresholds, never before NMS ranking.

**Consequence.** Calibration can change a verdict but cannot reorder detections. The report keeps both `raw_confidence` and calibrated `confidence`, so a reviewer can see the correction.

---

## ADR-006: Aggregate risk in addition to max confidence

**Context.** A vehicle with a severe crack and a vehicle with four moderate dents may both have a maximum confidence below a fail threshold, yet the second might still be unsafe. A max-confidence rule cannot express cumulative damage.

**Decision.** Compute `risk = Σ severity_weight × calibrated_confidence` and fail when risk reaches a configurable threshold. Also fail immediately when a severe class meets its own fail confidence.

**Consequence.** Two independent escalation paths exist, both explained in `reasons`. The threshold is an explicit policy parameter, not an implicit constant buried in code.

---

## ADR-007: Three verdicts instead of pass/fail

**Context.** Real inspection workflows have a human review queue. Forcing uncertainty into a binary verdict either releases questionable vehicles or blocks good ones.

**Decision.** Emit `pass`, `needs_review`, or `fail`. Treat the verdict as data; the CLI exits `0` for any completed inspection and `2` only for input/processing errors.

**Consequence.** Downstream systems can route review cases without parsing error messages. Operators can distinguish "the model is unsure" from "the request was malformed."

---

## ADR-008: Unknown classes fail closed

**Context.** A manifest/label mismatch or a new detector class should not silently disappear from an inspection.

**Decision.** `InspectionPolicy.inspect` raises `PolicyError` for labels without a rule. Detections below the confidence floor are retained in `ignored` with a reason; unknown classes are not.

**Consequence.** Missing policy coverage is a loud configuration error. Low-confidence noise is still auditable rather than discarded.

---

## ADR-009: Versioned report with an injected clock

**Context.** Reports are used by humans and machines. Unstable key order or hidden timestamps make diffs noisy and tests flaky.

**Decision.** `REPORT_SCHEMA_VERSION = "1.0"`, sorted JSON keys, and an optional `generated_at` parameter. Only the CLI reads the wall clock.

**Consequence.** The core is deterministic and testable. Consumers can reject unknown schema versions; Markdown is derived from the same dictionary, preventing drift between human and machine views.

---

## ADR-010: Preserve the research artifacts; be explicit about the model

**Context.** The original notebooks fine-tune ResNet on **CIFAR-10**, export to ONNX, and demonstrate Flask/FastAPI/Rust/WASM deployment. They are valuable evidence of cross-platform work, but they are not a vehicle-damage detector.

**Decision.** Keep every original file unchanged. Add the decision layer as a separate package. State the dataset and model limitation prominently in the README.

**Consequence.** The repository shows both research breadth and engineering depth without fabricating vehicle-specific metrics. A reviewer can evaluate the decision logic independently of the model's training data.

---

## ADR-011: CLI is a thin adapter

**Context.** Logic embedded in `argparse` handlers is hard to reuse and hard to test.

**Decision.** `cli.py` handles arguments, file reads, NMS orchestration, and output. All rules live in `geometry`, `calibration`, `policy`, `manifest`, and `report`. File errors are normalized to domain exceptions.

**Consequence.** A future HTTP endpoint can call the same modules. CLI tests exercise error paths without duplicating domain tests.

---

## ADR-012: No coverage exclusions for runtime branches

**Context.** A 100% badge achieved with `# pragma: no cover` on reachable branches is misleading.

**Decision.** No pragmas hide runtime logic. Accidental dead code is removed; genuine branches are tested. `__main__` is covered by both `runpy` execution and a normal import.

**Consequence.** The coverage number reflects executed behavior. The trade-off is a larger, more explicit test suite (372 cases), which is appropriate for a safety-adjacent decision layer.

---

## ADR-013: Manifest thresholds and safety policy are separate

**Context.** A model manifest can declare generic score/iou/review/fail thresholds, but a safety policy needs per-class severity. Applying one global review/fail pair to both a cosmetic scratch and a structural crack would either over-block vehicles or under-block dangerous defects.

**Decision.** Manifest `score` and `iou` drive NMS. Manifest `review` and `fail` are validated and reported as export-time model metadata. The CLI applies `policy.py` class rules, where each class owns its `min_confidence`, `review_confidence`, and `fail_confidence`.

**Consequence.** The model artifact remains portable and descriptive, while the safety decision stays explicit and reviewable in code. A future deployment can load a policy configuration with the same schema without changing the manifest contract.

---

## Deferred work

- Train and evaluate a detector on a real vehicle-damage dataset.
- Fit calibration temperatures on a held-out labelled set and publish reliability diagrams.
- Validate policy thresholds with domain experts and inspection standards.
- Add JSON Schema files for manifest and report, plus a compatibility test suite.
- Expose the same decision layer through a small HTTP service and a WASM client.
- Benchmark NMS and policy throughput on realistic detection volumes.
