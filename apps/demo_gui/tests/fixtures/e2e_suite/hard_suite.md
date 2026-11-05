# Full-run hard suite (GT report-only)

Stage 1 (both arms, large pulley) -> scripted UR open + exit -> stage 2 (Franka, second node) -> scripted release. Seated = large engaged at the stage-1 end; exit ok = no abort, Franka holding, large still engaged; success = both pulleys engaged after the release. * = run incomplete.

| arm | stage-1 seated | UR exit ok | stage 2 ran | both engaged | s1 solve p50/p95 med | s2 solve p50/p95 med | episode s med |
|---|---|---|---|---|---|---|---|
| V63j_tol50b | 16/16 | 16/16 | 16/16 | **7/16** | 11.7/25.0 | 22.4/34.8 | 38.6 |

Both engaged per start (stage-1 seated in brackets):

| start | V63j_tol50b |
|---|---|
| set2_gv_00 | 1/2 (2) |
| set3_gv_02 | 1/2 (2) |
| set3_gv_03 | 2/2 (2) |
| ur_z-30 | 1/2 (2) |
| up-20 | 0/2 (2) |
| up+20 | 0/2 (2) |
| up+30 | 0/2 (2) |
| nominal | 2/2 (2) |

Failure classes:

- V63j_tol50b: stage2:small over (engaged/over) x3, stage2:large off (slanted/outside) x2, stage2:large off (outside/outside) x1, stage2:large off (under/outside) x1, stage2:small slanted (engaged/slanted) x1, stage2:lost in the release (engaged/slanted) x1

| arm | start | ep | stage-1 end | stage-1 large | UR exit (large) | stage-2 end | final L/S | both | s1/s2 p95 ms | ep s | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V63j_tol50b | nominal | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/engaged | yes | 25.4/35.0 | 34.4 | success |
| V63j_tol50b | nominal | 1 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/engaged | yes | 23.6/30.4 | 34.2 | success |
| V63j_tol50b | set2_gv_00 | 0 | early_stop:engaged | engaged | ok (engaged) | completed | engaged/over | no | 25.3/31.1 | 54.1 | stage2:small over (engaged/over) |
| V63j_tol50b | set2_gv_00 | 1 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/engaged | yes | 24.7/35.5 | 34.3 | success |
| V63j_tol50b | set3_gv_02 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | outside/outside | no | 29.6/31.7 | 33.6 | stage2:large off (outside/outside) |
| V63j_tol50b | set3_gv_02 | 1 | early_stop:engaged | engaged | ok (engaged) | early_stop:engaged | engaged/engaged | yes | 29.7/30.8 | 38.2 | success |
| V63j_tol50b | set3_gv_03 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/engaged | yes | 24.4/34.0 | 42.1 | success |
| V63j_tol50b | set3_gv_03 | 1 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/engaged | yes | 29.2/34.6 | 34.8 | success |
| V63j_tol50b | up+20 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/over | no | 23.6/35.6 | 40.0 | stage2:small over (engaged/over) |
| V63j_tol50b | up+20 | 1 | early_stop:engaged | engaged | ok (engaged) | completed | slanted/outside | no | 25.9/35.6 | 54.4 | stage2:large off (slanted/outside) |
| V63j_tol50b | up+30 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | under/outside | no | 23.6/35.6 | 46.4 | stage2:large off (under/outside) |
| V63j_tol50b | up+30 | 1 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/slanted | no | 25.6/35.6 | 34.0 | stage2:small slanted (engaged/slanted) |
| V63j_tol50b | up-20 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | engaged/over | no | 24.4/33.1 | 44.5 | stage2:small over (engaged/over) |
| V63j_tol50b | up-20 | 1 | early_stop:engaged | engaged | ok (engaged) | early_stop:engaged | engaged/slanted | no | 24.3/35.6 | 40.9 | stage2:lost in the release (engaged/slanted) |
| V63j_tol50b | ur_z-30 | 0 | early_stop:engaged | engaged | ok (engaged) | abort:grasp_lost | slanted/outside | no | 27.2/30.0 | 37.0 | stage2:large off (slanted/outside) |
| V63j_tol50b | ur_z-30 | 1 | early_stop:engaged | engaged | ok (engaged) | early_stop:engaged | engaged/engaged | yes | 24.5/36.5 | 38.9 | success |
