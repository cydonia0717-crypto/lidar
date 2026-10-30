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

## Stage C: original ROS social-force population

The Noetic container builds copies of the author's `asr_rapidxml`, `ros_tools`, `pedsim_original`, `mpc_planner_msgs` and `pedestrian_simulator` packages. Build metadata corrections for isolated message generation are recorded as a patch. The original checkout is retained.

The added service wrapper compiles the original ROS simulator implementation by reference and invokes its public `Loop` and robot-state callback. It returns fresh original observation and prediction messages after each explicit world step. There is no free-running timer or artificial target trajectory substituted for the social-force engine.

The test uses the original 4/8/12-person corridor scenario files and fixed simulator seed5, 20Hz world steps and 5Hz controller updates. A separate Python risk-module copy changes only the hardcoded population5 to the input length. The bridge maps emitted future pose0 to future step1 and verifies every current predicted target reaches the objective. Original conversion failures and population truncation are recorded separately.

This remains an explicit compatibility adaptation: the ego plant is a bounded exact unicycle bridge, not Gazebo or Ackermann. It uses K64/T20/MC2048, a declared60s deadline and x>40m completion criterion. Collision checks are sampled at20Hz, with original checker radii0.325+0.3m and the scenario walls. Solve time is recorded but does not advance the lockstep world. It is not a paper-table reproduction, real-time or continuous-safety certificate, real LiDAR result, or a new algorithm.

## Following stages

1. Trace the original launch, prediction callbacks and configuration; resolve/document paper-versus-code differences.
2. Build a headless ROS environment, run the original controller and save navigation traces with separate success/collision/timeout/error outcomes.
3. Reproduce representative original comparisons before making an Ackermann adaptation.
4. Select an extension only after failures are repeatable and simple/strong controls cannot explain an improvement.

This workflow intentionally does not launch the author's privileged development container or connect to physical actuators.

## Stage D: frozen development comparison and failure diagnostics

`protocol-stage-d.json` declares 12 paired development episodes (mean-only vs DRA, 4/8/12 people, base seeds5/17), plus6 known-case diagnostics: no-wall for all three populations, and no-filter / K128 / exact-arc prediction for8 people at seed5. It is frozen before execution. These are not unseen validation trials.

Both main methods share K64/T20, the same exact unicycle plant, explicit output limits, and a20000 finite wall penalty at |y|>=3.175m. DRA additionally uses MC2048. Same candidate counts do not mean equal wall-clock budgets. CPU solve times are measured with the world paused.

The deterministic mean-only adapter follows the paper's baseline description, but its unpublished exact weights are unavailable. It uses20000 per predicted step if any mean is within0.825m, and keeps the author's goal/path/speed costs, Halton sampler, MPPI update and output filter. It must not be labelled the author's exact Vanilla MPPI reproduction. The original DRA risk expression is retained.

The harness verifies paired initial observation/prediction hashes, checks the adapters with independent fixtures, records per-episode wall/pedestrian collision types, path length and waiting, and audits the unfiltered/filtered averaged output against exact-arc wall margins without consuming planner RNG state. Finite penalties, weighted outputs and sampled collision checks do not guarantee safety. No compatibility repair or pre-existing exact arc integration is claimed as a research contribution.

### Stage D2 collision-population correction

Run40999340073 retains the original all-observation checker. Forensic spatial matching found8 of9 pedestrian-contact terminations involved entities excluded from active predictions. Prediction IDs are renumbered after completed agents are skipped, so ID equality alone cannot establish association. The author's original collision checker subscribes to `trajectory_predictions`, whereas `GetPedestrians()` also emits completed entities. D2 reruns the same18 frozen variants with the author's active-population primary checker and separately records every historical-object gap/contact. It does not relabel previously stopped trajectories as successes or claim contacts are safe for real humans. All-observation contact times and counts must accompany primary outcomes.

## Stage E: coherent physical population and finite-budget risk diagnostics

The original-active D2 protocol remains separately runnable. New physical protocol E retains all internal simulator entities in both Gaussian predictions and collision checks, including completed entities. Only the done_ skip in a build-only copy of GetGaussianPredictions is removed; current IDs and positions must match observations at every20Hz world step.18 development episodes compare mean-only, original DRA risk, and an exact independent isotropic-Gaussian event-CDF baseline, sharing wall/actuation conditions and K64/T20. The event-CDF baseline is existing math, not an original contribution.

The separately frozen risk protocol contains28 banks/224 distinct threshold-balanced constructed queries,20 resampling seeds, four budgets512/2048/8192/20000, original expression, event-consistent position Monte Carlo, fixed exact-binomial confidence, sequential confidence and deployable analytic CDF baselines. Development and constructed held-out parameter combinations are separated; neither is navigation-map validation. Exact CDF is cross-checked by independent polar quadrature, and original-expression quadrature separates structural differences from MC noise.

The sequential baseline spends family alpha across8 fixed queries and4 predeclared looks, so shared randomness and selecting a certified candidate do not require independent queries. Claims still require iid draws from the declared model and fixed query banks, and exclude misspecified predictions, unbounded adaptive queries and whole-mission safety. Unresolved cases remain unresolved, not safe. Promotion gates use development only and require benefit beyond deployable analytic baselines before an adaptive controller is introduced. These are strong baselines and an experimental scaffold, not a claimed new algorithm.
