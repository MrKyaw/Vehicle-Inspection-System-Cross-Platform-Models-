# About

**Vehicle Inspection System — Cross-Platform Models** is a research-to-engineering upgrade of a computer-vision deployment prototype.

The original work explored the cross-platform path of a PyTorch image classifier: fine-tune a ResNet on CIFAR-10, export it to ONNX, run inference with ONNX Runtime, serve it through Flask and FastAPI, preprocess images in Rust/WebAssembly, and run the model in the browser with ONNX Runtime Web.

This repository adds `vehicle_inspection`, a **dependency-free Python 3.10+ package** that models the operational decision layer for vehicle inspection:

- detection box geometry, IoU, deterministic non-maximum suppression, letterbox resize and inverse mapping;
- strict JSON model-manifest validation across ONNX, PyTorch, and WASM deployment targets;
- per-class temperature calibration, expected calibration error, and confidence buckets;
- a safety-oriented policy that produces `pass` / `needs_review` / `fail` with explicit reasons;
- a versioned JSON report schema, Markdown renderer, and `python -m vehicle_inspection` CLI.

## What it is

- A reviewer-facing demonstration of domain modeling, computer-vision fundamentals, deterministic algorithms, API/CLI design, and test discipline.
- A runnable, offline reference implementation with **372 tests** and **100% line and branch coverage** of the new package.
- An additive layer that preserves the original notebooks, Rust crate, server samples, ONNX export, screenshot, and MIT license.

## What it is not

- Not a trained vehicle-damage detector. The research model is a CIFAR-10 classification export, not a vehicle dataset model.
- Not a certified or regulated inspection product.
- Not a benchmark result. The reported numbers are test and coverage counts from actual local runs.

## Topics and keywords

`vehicle-inspection`, `computer-vision`, `object-detection`, `bounding-boxes`, `non-maximum-suppression`, `iou`, `letterbox`, `model-manifest`, `onnx`, `onnx-runtime`, `pytorch`, `webassembly`, `rust`, `flask`, `fastapi`, `confidence-calibration`, `temperature-scaling`, `expected-calibration-error`, `decision-policy`, `safety-critical`, `json-schema`, `reporting`, `cli`, `pytest`, `branch-coverage`, `ci`, `python`, `standard-library`, `offline`, `research-prototype`, `portfolio`.

## Author

Kyaw Kyaw Oo — [kyaw.tech@gmail.com](mailto:kyaw.tech@gmail.com) — [github.com/MrKyaw](https://github.com/MrKyaw)

## License

MIT. See [License.text](License.text).
