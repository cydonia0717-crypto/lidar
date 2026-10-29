# DRA-MPPI cloud reproduction, stage A

Foundation: Trevisan et al., Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations, IROS 2025, DOI 10.1109/IROS60139.2025.11246822.

- Paper: https://arxiv.org/abs/2506.21205v2
- Original implementation: https://github.com/KhMustafa/dra-mppi
- Pinned public revision: `6d76ab2878c503121544b992c0e5706c6c6dece0` (not yet confirmed as the paper-time revision).

## What runs here

GitHub Actions checks out the original implementation separately and executes its **unchanged** `JackalDynamics` and `DynamicObstacles` modules on a CPU runner. The repository contains our runner and reference calculations, not a copy of the upstream project.

The fixed fixtures exercise straight/turning propagation, a single isotropic Gaussian disk probability against a noncentral-chi-square reference, and independent multiple-obstacle batch diagnostics. Source hashes, active configuration, environment, raw values and findings are saved as artifacts.

Multi-obstacle diagnostics compare the source output with a reference that integrates each obstacle event before taking the independent union. Deviations are reported without rewriting the author's estimator. A successful workflow means the execution assertions passed; it does not assert that all probability diagnostics agree or that the controller is safe.

## Scope boundaries

- These are constructed module inputs, **not** navigation episodes, unseen navigation validation, real LiDAR processing, or real-car tests.
- The ROS/Gazebo loop, full MPPI controller, original result tables, multi-mode ROS callbacks, Ackermann adaptation and GPU timing are **not yet reproduced**.
- The original constructor hardcodes five obstacles. The batch fixtures use its direct Gaussian API; this does not validate the callback path or fix the hardcoded bookkeeping.
- Covariance convention is explicit: fixtures pass diagonal standard-deviation matrices because the original module squares the input elementwise.
- Original risk threshold: a stagewise model quantity, not an unconditional whole-mission safety guarantee. Paper Remark 1 already discusses the risk of the weighted output plan.

## Run

The branch `research/dra-mppi-cloud` triggers `.github/workflows/dra-mppi-upstream.yml` on push. It uses the standard CPU runner, no self-hosted runner or purchased GPU.

With Python 3.10, the pinned upstream checkout and requirements installed:

```bash
python research/dra_mppi/run_upstream_audit.py --upstream /path/to/dra-mppi --output results/dra-mppi
```

The workflow installs PyTorch 2.4.1 CPU separately using the official PyTorch package index.

## Following stages

1. Trace the original launch, prediction callbacks and configuration; resolve/document paper-versus-code differences.
2. Build a headless ROS environment, run the original controller and save navigation traces with separate success/collision/timeout/error outcomes.
3. Reproduce representative original comparisons before making an Ackermann adaptation.
4. Select an extension only after failures are repeatable and simple/strong controls cannot explain an improvement.

This workflow intentionally does not launch the author's privileged development container or connect to physical actuators.
