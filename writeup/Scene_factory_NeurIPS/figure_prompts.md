# SceneFactory — Figure Generation Prompts for AI Diagram Tools

> **Purpose**: Each prompt below is a self-contained, copy-paste-ready instruction
> for an AI figure generation model (GPT-4o image gen, Gemini, Claude artifacts,
> Midjourney, etc.) to produce a publishable scientific illustration.
>
> **Style target**: NeurIPS / ICML / CoRL / T-ITS publication quality.
> Clean vector-style diagrams, muted colorblind-safe palette, serif labels,
> no gradients or 3D chrome. Think: figures in Dreamer-v3, GPUDrive (ICLR 2025),
> Decision Transformer, Isaac Lab papers.

---

## FIGURE 1 — System Architecture / Pipeline Overview  (`fig:architecture`)

> **This is the most important figure in the paper.**

```
Create a clean, publication-quality horizontal pipeline diagram for a NeurIPS
2026 paper titled "SceneFactory: GPU-Accelerated Multi-Agent Driving Simulation
with Physics-Based Vehicle Dynamics". The figure should span the full text width
(~5.5 inches) and be roughly 2.5 inches tall.

LAYOUT: A left-to-right horizontal flow with 4 main stages connected by
right-pointing arrows. Each stage is a rounded-rectangle box with a short title
in bold and 2-3 bullet keywords inside in smaller text.

THE 4 MAIN STAGES (left to right):

  Stage 1 — "Data Ingestion" (light gray background)
    Icon suggestion: a database cylinder or file icon
    Bullets inside the box:
      • Waymo Open Motion Dataset TFRecords
      • Per-scenario JSON extraction
      • Road polylines + agent start/goal poses

  Stage 2 — "Procedural Scene Construction" (light blue background)
    Icon suggestion: a grid/mesh icon
    Bullets inside the box:
      • Z-flattening, re-centering
      • USD road prims (PointInstancer)
      • Multi-world 2D grid layout (N worlds × 400 m pitch)
      • GPU metadata tensor baking

  Stage 3 — "Environment Instantiation" (light green background)
    Icon suggestion: multiple overlapping rectangles (cloned envs)
    Bullets inside the box:
      • Isaac Lab clone + post-clone road injection
      • 10-DOF articulated PhysX vehicles (up to M=16/world)
      • Per-world friction from weather module

  Stage 4 — "GPU-Resident Training Loop" (light orange background)
    Icon suggestion: a circular arrow or GPU chip icon
    Bullets inside the box:
      • Shared-policy PPO via RSL-RL
      • N×M agent slots (e.g. 256×16 = 4,096)
      • All inference + gradient updates on-device
      • 30 Hz control / 120 Hz physics

TWO AUXILIARY MODULE CALLOUT BOXES feeding into Stage 3 from below:

  Callout A — "Vehicle Sysid" (light purple, smaller box, dashed border)
    Connected to Stage 3 with an upward arrow
    Contents: "CEM-based parameter fitting", "20 tunable dynamics params",
    "7 scripted maneuvers vs. PhysX teacher"

  Callout B — "Weather-to-Friction" (light teal, smaller box, dashed border)
    Connected to Stage 3 with an upward arrow
    Contents: "Precipitation → water film thickness h_w",
    "Modified LuGre model (Zhao et al.)", "μ_eff → PhysX material + obs"

ARROWS between stages: simple right-pointing arrows with small labels on them:
  Stage 1 → Stage 2: label "JSON"
  Stage 2 → Stage 3: label "USD stage"
  Stage 3 → Stage 4: label "Tensor obs/act"

STYLE:
  - Flat design, no 3D effects, no gradients, no drop shadows
  - Muted pastel fill colors, thin dark gray borders (1 pt)
  - Font: serif (Times New Roman style), all text black or dark gray
  - Bold stage titles (~11 pt), regular bullets (~8 pt)
  - Arrows: dark gray, 1.5 pt stroke, filled arrowhead
  - White or very light background
  - The overall look should match figures in top ML venues (NeurIPS, ICML, ICLR)
    and transportation journals (T-ITS, TRC)
  - Colorblind-safe palette: use Wong (2011) muted tones
  - No decorative elements — every pixel should convey information
```

---

## FIGURE 2 — Neural Network Architecture  (`fig:network_arch`)

```
Create a clean, publication-quality neural network architecture diagram for a
NeurIPS paper. The network is a "late-fusion actor-critic" with per-modality
encoders and max-pool set aggregation.

LAYOUT: Top-to-bottom dataflow diagram, approximately 3.25 inches wide and
4 inches tall (half-column width for a NeurIPS paper).

TOP ROW — Three parallel INPUT boxes (horizontally arranged):

  [Ego State]          [Road Points]           [Neighbor Vehicles]
  dim: 11              dim: 350 × 5            dim: 24 × 7
  (gray box)           (blue box)              (orange box)

  Small annotation under each: the contents
  Ego: "goal pos, heading err, velocity, weather"
  Road: "K_r=350 nearest pts within 10m"
  Vehicle: "K_v=24 nearest alive agents + TTC"

SECOND ROW — Three parallel ENCODER MLPs (one per input):

  [Ego MLP]            [Road MLP]              [Vehicle MLP]
  11→64→64             5→96→96                 7→96→96
  (same color as       (same color as          (same color as
   input above)         input above)            input above)

  Each MLP box shows "ELU" activation label inside.
  The Road and Vehicle MLPs have a small annotation: "applied per-entity"

THIRD ROW — Aggregation step (only for Road and Vehicle branches):

  The Ego MLP output passes straight down (64-dim arrow).

  The Road MLP outputs go into a [Masked Max-Pool] box → 96-dim output.
  Small note: "zero-padded slots → -∞"

  The Vehicle MLP outputs go into a [Masked Max-Pool] box → 96-dim output.
  Same note.

FOURTH ROW — Concatenation:

  A horizontal bar or junction symbol merges the three streams:
  64 + 96 + 96 = 256-dim
  Label this junction "Concatenate (256)"

FIFTH ROW — Shared Trunk MLP:

  [Trunk MLP: 256→128→64, ELU]
  (neutral gray box)

SIXTH ROW — Two output heads branching left and right:

  Left branch: [Actor Head]
    "→ 3-dim μ" (mean of Gaussian)
    "learnable log σ"
    "Diagonal Gaussian policy"
    Output: "throttle, steering, brake"
    (green box)

  Right branch: [Critic Head]
    "→ 1-dim V(s)"
    "State value estimate"
    (purple box)

ANNOTATION at the very top or in a small legend box:
  "Actor and critic do NOT share encoder weights — separate MLPs for each."

STYLE:
  - Flat design, no 3D, no gradients
  - Rounded rectangles for all boxes
  - Arrows: thin dark gray lines with small arrowheads
  - Dimension annotations (e.g., "64-dim") on arrows or next to them in small text
  - Muted colorblind-safe colors: use the same color per modality (gray=ego,
    blue=road, orange=vehicle, green=actor, purple=critic)
  - Serif font (Times New Roman), black text
  - Matches NeurIPS / ICML figure conventions
```

---

## FIGURE 3 — Waymo-to-USD Scene Construction Pipeline  (`fig:scene_pipeline`)

```
Create a publication-quality horizontal pipeline diagram showing how a single
Waymo driving scenario is converted into a GPU-ready USD road environment.
This is a "mini-figure" for a NeurIPS paper, approximately 5.5 inches wide
and 1.8 inches tall.

LAYOUT: 5 stages flowing left to right, each represented as a rounded box
with a small representative illustration ABOVE it and a text description
BELOW. Stages are connected by right-pointing arrows.

STAGE 1 — "Raw Waymo TFRecord"
  Illustration above: a small schematic of scattered 2D polyline points in
  different colors (blue dots for lane centers, red dots for road edges),
  with a few vehicle icons (small rectangles) at start/goal positions.
  Looks like a raw bird's-eye view of an intersection — messy, with 3D
  elevation variation.
  Label below: "~30K road points\n128 agent tracks\n91 timesteps"

  → Arrow labeled "extract & group"

STAGE 2 — "Per-Scenario JSON"
  Illustration above: a clean JSON-like text snippet:
    { "road": { "polylines": [...] },
      "agents": { "items": [...] } }
  But stylized, not literal code — just a hint of the structure.
  Label below: "Typed polylines (PCA-ordered)\nAgent start/goal poses"

  → Arrow labeled "build world"

STAGE 3 — "Re-centered Road Segments"
  Illustration above: a cleaner bird's-eye view with road segments shown as
  oriented rectangles, centered at (0,0). Blue segments = lane centers,
  red segments = road edges. The scene is z-flattened (flat). Segments are
  aligned end-to-end forming road lanes. A small coordinate axis at center.
  Label below: "Z-flatten, re-center\nOriented box segments\nBounding box filter (±200m)"

  → Arrow labeled "USD prims"

STAGE 4 — "USD PointInstancer"
  Illustration above: same road layout but now shown as a clean USD scene
  with colored segments. A small "customData" annotation box shows:
  "positions[], directions[], types[], half_widths[]"
  Representing the baked metadata arrays.
  Label below: "One cube prototype × N instances\nMetadata baked to customData\nGPU-friendly flat arrays"

  → Arrow labeled "tile N×"

STAGE 5 — "Multi-World Grid Stage"
  Illustration above: a bird's-eye view of a 4×4 grid (or smaller like 3×3)
  of different road scenarios, each in its own cell, separated by spacing.
  Each cell shows a slightly different road layout (intersection, curve,
  straight). Small colored dots represent agent spawn positions in each world.
  Label below: "N worlds on 400m grid\nHeterogeneous topologies\nUp to M=16 agents/world"

STYLE:
  - Scientific illustration style, flat colors, no photorealism
  - The bird's-eye views should be SCHEMATIC — simple colored lines and
    rectangles representing roads, not photorealistic renderings
  - Use blue for lane-center geometry, red/pink for road-edge geometry,
    green dots for agent start positions, yellow/gold stars for goal positions
  - Muted pastel backgrounds in each stage box (alternate light gray / white)
  - Thin dark borders, serif text labels
  - Arrow labels in italic, 7pt font
  - Overall look: clean, information-dense, matches NeurIPS style
```

---

## FIGURE 4 — 10-DOF Articulated Vehicle  (`fig:vehicle_structure`)

```
Create a publication-quality technical diagram showing the kinematic structure
of a 10-DOF articulated rigid-body vehicle for a NeurIPS robotics paper.

LAYOUT: Side-by-side with (a) a schematic 3/4-view of the vehicle and
(b) a kinematic tree diagram. Total width ~5.5 inches, height ~2.5 inches.

PANEL (a) — "Vehicle Schematic" (left, ~3 inches wide):

  A simplified 3/4-view (slightly elevated perspective) of a sedan-like box
  vehicle showing:

  - A rectangular box CHASSIS (4.0 × 2.0 × 1.0 m, 1800 kg) in light gray,
    slightly transparent so internal structure is visible.
  - Four WHEELS at the corners (cylinders, radius 0.35m, width 0.15m) in
    dark gray.
  - At each wheel corner, show the JOINT CHAIN with colored annotations:
      • PRISMATIC joint (Z-axis, ±0.175m travel): shown as a vertical
        double-headed arrow in BLUE at each corner
      • REVOLUTE joint (Z-axis, ±32°): shown as a curved arrow in GREEN
        at the two FRONT corners ONLY (not rear)
      • CONTINUOUS joint (Y-axis, wheel spin): shown as a circular arrow
        in ORANGE around each wheel
  - Label the wheelbase (2.6m) with a horizontal dimension line between
    front and rear axles.
  - Label the track width (2.0m) with a horizontal dimension line between
    left and right wheels.
  - Small text annotation: "Front-wheel drive, front-wheel steering"
  - Mark the chassis center of mass with a small cross-hair symbol.

PANEL (b) — "Kinematic Tree" (right, ~2.5 inches wide):

  A tree diagram showing the articulated body hierarchy:

  [Chassis (root)]
    ├── [FL Suspension] ─ prismatic Z
    │   └── [FL Steering] ─ revolute Z
    │       └── [FL Wheel] ─ continuous Y
    ├── [FR Suspension] ─ prismatic Z
    │   └── [FR Steering] ─ revolute Z
    │       └── [FR Wheel] ─ continuous Y
    ├── [RL Suspension] ─ prismatic Z
    │   └── [RL Wheel] ─ continuous Y
    └── [RR Suspension] ─ prismatic Z
        └── [RR Wheel] ─ continuous Y

  Each node is a small rounded rectangle. Joint labels are on the edges.
  Color-code joints to match panel (a): blue=prismatic, green=revolute,
  orange=continuous.
  Note that rear corners have NO steering joint (only 2 joints instead of 3).

STYLE:
  - Clean technical illustration, no photorealism
  - Panel labels "(a)" and "(b)" in top-left corners
  - The vehicle in (a) should be a simple geometric box-and-cylinder model,
    NOT a realistic car rendering
  - Thin annotation lines and arrows, serif labels
  - White background, thin black panel borders
  - Matches engineering/robotics paper figure conventions
  - Color-coded joints with a small legend: Blue=Suspension(prismatic),
    Green=Steering(revolute), Orange=Spin(continuous)
```

---

## FIGURE 5 — Weather-to-Friction Module Pipeline  (`fig:friction_pipeline`)

```
Create a clean, publication-quality diagram showing a weather-to-friction
module pipeline for a NeurIPS paper on autonomous driving simulation.
Width ~3.25 inches (half-column), height ~3.0 inches.

LAYOUT: A vertical flow diagram with 3 stages, plus a small inset plot.

TOP — Input block:
  Two input boxes side by side:
    [Precipitation Level]     [Road Surface Type]
    "water film thickness     "AC / SMA / OGFC"
     h_w (0-1.0 mm)"          (one-hot encoded)
  Both feed downward arrows into:

MIDDLE — Model block (main box, prominent):
  A larger rounded box labeled "Modified Average Lumped LuGre Model"
  with subtitle "(Zhao et al., 2024)"
  Inside, show the key equation in LaTeX-style math:
    μ_eff = max(θ·Y_R − Y_F, 0) · θ·Y_R·σ₀·z̄*
  Below the equation, small text: "Stribeck friction + hydrodynamic lift"

  This box has a downward arrow splitting into TWO outputs:

BOTTOM — Two output paths:

  Left output path → [PhysX Ground Material]
    Box content: "μ_static, μ_dynamic"
    "Applied to global ground plane"
    Note: "(uniform across cloned envs)"
    (This path has a small "⚠" icon indicating a current limitation)

  Right output path → [Policy Observation]
    Box content: "o_weather = [h_w/h_norm, 1_AC, 1_SMA, 1_OGFC] ∈ ℝ⁴"
    "Appended to ego state (11-dim total)"
    (This path has a "✓" checkmark indicating this is the active path)

INSET PLOT (small, tucked into the top-right corner or bottom, ~1.5 × 1.0 inches):
  A mini line plot showing μ_eff vs. h_w (water film thickness, 0 to 1.0 mm)
  for three road surface types:
    - AC (Asphalt Concrete): lowest curve, labeled, in blue
    - SMA (Stone Mastic Asphalt): middle curve, in orange
    - OGFC (Open-Graded Friction Course): highest curve, in green
  All curves decrease from left to right (higher water = lower friction).
  X-axis: "Water film thickness h_w (mm)"
  Y-axis: "μ_eff"
  Title: "@ v = 13.89 m/s"

STYLE:
  - Flat design, muted colors, no gradients
  - Box fills: light gray for inputs, light blue for model, light green/red
    for outputs
  - Serif font, thin borders
  - The inset plot should have minimal axes (no top/right spines), matching
    the data-plot style conventions of the same paper
  - Matches NeurIPS figure conventions
```

---

## FIGURE 6 — Multi-Agent Environment Vectorization  (`fig:vectorization`)

```
Create a publication-quality conceptual diagram illustrating the key
architectural difference between SceneFactory and a conventional (non-vectorized)
driving simulator. This is the diagram that explains the 127× speedup.
Width ~5.5 inches (full column), height ~2.5 inches.

LAYOUT: Two panels side by side, labeled "(a) Baseline: Per-Agent Python Loops"
and "(b) SceneFactory: Batched GPU Tensor Operations".

PANEL (a) — Baseline (left, ~2.5 inches):

  Show a CPU box at the top and a GPU box at the bottom.

  CPU box contains a FOR LOOP visualization:
    "for agent_i in range(N×M):"
    Inside the loop body, show 4 sequential steps in a vertical stack:
      [1. Query state_i from GPU]  ← small arrow going CPU↔GPU
      [2. Compute obs_i]
      [3. Compute reward_i]
      [4. Send action_i to GPU]    ← small arrow going CPU↔GPU
    A red "bottleneck" annotation or hourglass icon on the CPU↔GPU arrows.

  GPU box contains:
    [PhysX Solver] — "runs fast"
    But has idle/waiting indicators showing it's underutilized.

  Below panel: "~152 CASPS"

PANEL (b) — SceneFactory (right, ~2.5 inches):

  Show a single GPU box (no CPU involvement in the inner loop).

  Inside the GPU box, show 4 PARALLEL/BATCHED steps in a vertical stack,
  but each step has a wide bar indicating ALL agents processed simultaneously:
    [1. Batched state query — torch tensor (N×M, d)]
    [2. Batched obs assembly — torch tensor ops]
    [3. Batched reward computation — torch tensor ops]
    [4. Batched action decode — torch tensor ops]
  Each bar is wide and has a small grid pattern suggesting many agents in parallel.

  Below these 4 steps:
    [PhysX Solver — parallel broad-phase/narrow-phase]

  Below these:
    [RSL-RL PPO — on-device gradient updates]

  An annotation: "Zero CPU↔GPU transfers in inner loop"

  Below panel: "~19,250 CASPS (127×)"

CONNECTING ELEMENT between panels:
  A large "×127" or "127× faster" annotation between the two panels,
  possibly with a right-pointing arrow or lightning bolt.

STYLE:
  - Schematic / conceptual — not a code diagram
  - Panel (a) should look slow/sequential (use warm red/orange tints,
    hourglass icons, thin serial arrows)
  - Panel (b) should look fast/parallel (use cool blue/green tints,
    wide parallel bars, GPU chip icon)
  - Flat design, serif labels, muted colors
  - The contrast between sequential and parallel should be immediately
    visually obvious
  - Matches ML systems paper figure conventions (like those in Madrona,
    Isaac Lab, EnvPool papers)
```

---

## FIGURE 7 — Observation Space Schematic  (`fig:observation`)

```
Create a publication-quality bird's-eye-view diagram showing the observation
space of a single ego agent in the SceneFactory driving simulator.
Width ~3.25 inches (half-column), height ~3.25 inches.

LAYOUT: A top-down (bird's-eye) view of a road scene centered on the ego
vehicle, with annotations showing all three observation groups.

CENTRAL ELEMENT — Ego Vehicle:
  A small dark-blue rectangle (representing a car from above, ~4.0×2.0m)
  at the center of the figure, pointed upward (forward direction).
  A small coordinate frame at the vehicle center: x-axis (forward, up on page),
  y-axis (left).
  Below the car, a small label: "Ego state (11-dim): goal, heading err,
  velocity, weather"

OBSERVATION GROUP 1 — Road Geometry (K_r = 350 points):
  Draw road lanes as pairs of parallel gray lines forming a road network
  (e.g., an intersection or curved road) around the ego vehicle.
  Within a 10-meter radius circle (drawn as a dashed circle centered on ego):
    Show scattered small BLUE DOTS on the road lanes — these represent the
    350 nearest road-segment midpoints.
    Some dots should be on lane centers, some on road edges.
  Outside the 10m circle: road is still visible but dots are absent.
  Annotation box: "Road context: K_r=350 nearest points within r=10m"
  "Per point: (Δx, Δy, type, dir_x, dir_y)"

OBSERVATION GROUP 2 — Neighbor Vehicles (K_v = 24):
  Scatter 6-8 other vehicle rectangles (smaller, in ORANGE) at various
  positions around the ego on the road lanes. Some ahead, some behind,
  some in adjacent lanes.
  Draw thin dashed lines from ego to each neighbor vehicle.
  On one or two of these lines, show a "TTC" annotation with a small
  swept-circle visualization (three overlapping circles along each vehicle's
  length axis).
  Annotation box: "Neighbors: K_v=24 nearest alive agents"
  "Per neighbor: (Δx, Δy, L, W, Δψ, speed, TTC)"

GOAL MARKER:
  A gold/yellow STAR somewhere ahead along the road, representing the
  goal position.
  A thin arrow from ego toward the star, labeled "d_goal".
  The heading error angle Δψ shown as a small arc between the ego's
  forward direction and the direction to the goal.

STYLE:
  - Clean schematic bird's-eye view, not photorealistic
  - Road drawn as simple parallel lines (light gray fill, darker edges)
  - Color coding: blue=road observation points, orange=neighbor vehicles,
    dark blue=ego, gold=goal
  - Dashed circle for the 10m observation radius
  - Small, clean annotation boxes with thin borders
  - Serif font, matches NeurIPS conventions
  - The diagram should be information-dense but not cluttered
```

---

## Usage Notes

1. **For GPT-4o / ChatGPT image generation**: Copy the prompt inside the ``` block directly. Add "Output as a clean SVG-style scientific figure on a white background." at the end.

2. **For Midjourney**: Prepend `Scientific diagram, technical illustration, flat design, NeurIPS paper style --ar 16:9 --v 6` and condense the prompt to key elements.

3. **For Claude artifacts / Gemini**: Copy the full prompt as-is. These models handle detailed structured instructions well.

4. **For TikZ / Excalidraw**: Use the prompt as a specification document and manually build the figure, or ask an LLM to generate TikZ code from the prompt.

5. **Post-processing**: After AI generation, refine in Inkscape/Illustrator to ensure:
   - Fonts are embedded Times New Roman or Computer Modern
   - Colors exactly match the paper's colorblind-safe palette
   - Text is editable (not rasterized)
   - Export as PDF with `fonttype=42` for NeurIPS submission
