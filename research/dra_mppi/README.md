# DRA-MPPI cloud reproduction, stages A and B

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
- Stage A does not run the full controller. Stage B adds unmodified upstream MPPI, objective and dynamics in a custom headless harness, with one paper-sized planning call and reduced-budget short closed loops. This is **not** the ROS/Gazebo loop, original result tables, multi-mode ROS callbacks, Ackermann adaptation or GPU timing.
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

## Stage B

`crosscheck_risk.py` compares the original output against three independent event calculations: noncentral chi-square CDF, polar quadrature and direct target-position sampling. It separately integrates the density expression used by the original implementation. The fixed inputs include separated target means and the earlier overlapping-ring stress fixture. Threshold disagreements are diagnostics, not collision episodes or an established contribution.

`run_controller_smoke.py` calls the original `MPPIPlanner`, `JackalObjective`, `JackalDynamics`, `DynamicObstacles`, spline and Halton sampler. It does not patch or replace them. Five targets match the original constructor's hardcoded bookkeeping. A single 400x20x20000 planning call checks paper-sized allocation. Six custom short episodes use 64 MPPI samples, 20 prediction steps and 4096 Monte Carlo points, on CPU.

The independent plant is an exact constant-speed/yaw-rate unicycle arc. Targets follow declared fixed-velocity paths. The 4m navigation task, 12s deadline and radius0.825m collision check are explicitly custom. Success, collision, timeout, execution error and unresolved event checks remain separate. Source output is not silently clipped after averaging/filtering; output-bound deviations are logged. No comparison against another method is claimed.

Build-only setup for the author's `ghalton==0.6.1` uses Clang and libc++ in the disposable runner; no sampler compatibility shim is substituted.

## Following stages

1. Trace the original launch, prediction callbacks and configuration; resolve/document paper-versus-code differences.
2. Build a headless ROS environment, run the original controller and save navigation traces with separate success/collision/timeout/error outcomes.
3. Reproduce representative original comparisons before making an Ackermann adaptation.
4. Select an extension only after failures are repeatable and simple/strong controls cannot explain an improvement.

This workflow intentionally does not launch the author's privileged development container or connect to physical actuators.
