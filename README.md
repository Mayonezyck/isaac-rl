# isaac-rl

## currrent working vehicle optimization

`nohup /home/yz8733/miniforge3/envs/isaac/bin/python -m src.student_vehicle_sysid   --headless   --teacher-dataset-manifest artifacts/physx_teacher_datasets/fwd_v1/manifest.json   --student-usd artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd   --output-dir artifacts/student_vehicle_sysid/fwd_v1_staged_cem_overnight_02   --search-mode staged   --optimizer cem   --random-search-trials 640   --random-search-seed 123   --cem-population-size 32   --cem-elite-fraction 0.20   --cem-initial-std-fraction 0.20   --cem-min-std-fraction 0.02   > artifacts/student_vehicle_sysid/fwd_v1_staged_cem_overnight_02.log 2>&1 &`