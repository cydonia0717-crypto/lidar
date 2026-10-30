# Frozen Stage E baseline, 2026-10-30

Frozen source: d252bcfdc13f47dd7de8e109f82663ecb44c0844. Run [41002729726](https://github.com/cydonia0717-crypto/lidar/actions/runs/41002729726), all four jobs successful. Stage E ZIP SHA256 c7c60d719c66d96be19e63e8c0210b70cd191470e2b1332c094f47bc431f46a7. Independent replay: evidence/stage_e_replay.json. Original artifacts remain unchanged, no D2/E pooling.

| Item | Paper / fixed public source | Stage E |
|---|---|---|
| Vehicle | Jackal/unicycle | Exact unicycle ego; original batch prediction; no Ackermann |
| Planning budget | K400,T20,dt0.2,MC20000 | K64,T20,dt0.2,MC2048; CDF has no MC |
| Compute | Paper GPU/CPU hardware | GitHub CPU, paused lockstep world; not equal time budgets |
| Population | Public prediction skips done_, observations retain them | Build copy predicts all retained physical objects; same collision population |
| Collision | Public checker consumes active predictions | All entities plus walls,20Hz,physical radius sum0.625,planning buffer0.825 |
| Prediction | Paper Gaussian and multimodal scenarios | Unimodal CV from simulator current state,constant factor0.3 |
| Observation | No claim of perception reproduction | Privileged current simulator state; no future-realization access found |
| Task | Paper corridor experiment | x>40 before60s; collision priority; not goal stopping |
| Baselines | SH-MPC,Frenet,Vanilla in paper | Our mean-only adapter, original DRA risk,CDF adapter |
| Statistics | Paper100-trial comparisons | Two environment seeds, six paired conditions per method,development only |

Outcomes S/C/T: mean5/1/0,DRA4/2/0,CDF5/1/0. Paper DRA trend of superior success not reproduced at these conditions; this does not disprove the paper. CDF is faster on the constructed analytic family, not a new algorithm or uniformly better navigator.

Common settings: same initial observations/predictions verified by hashes; same walls, plant, clipping, task and sampler budgets. Differences in risk cost are intentional method definitions; equalK is not equal runtime. Original source preserved separately, count compatibility and physical population patches recorded. No failures omitted:18 episodes,0 execution errors. Prediction-position correspondence rechecked at2975 logged control instants;20Hz identity assertions are runner evidence, not an independent saved20Hz prediction replay. Collision replay at11947 observation frames agrees; between-frame contact is not excluded.

Stage F is a separately declared synthetic, scripted-world diagnostic using the same upstream planner. It will not change these evaluators or call its results paper-table reproduction. F2 predictor diagnostic variants may use known futures only when explicitly labelled oracle, never as a deployable method.
