# Stage 1 Navigation Safety

This stage-1 preset is the safety-aware variant of the current 4-world curriculum.

It keeps the easier stage-1 map scale:

- 4 worlds
- 8 max agents per world
- dry / clear `AC`

but turns on the task signals that matter for disciplined driving:

- road-edge termination
- vehicle-contact termination
- lane-center reward
- solid-line penalty

This makes it the right base for hyperparameter search when you want the search objective to care about:

- `rollout/road_contact_done_rate`
- `rollout/vehicle_contact_done_rate`

Use this as a short-horizon safety-aware stage-1 search base before promoting the winner into longer training.
