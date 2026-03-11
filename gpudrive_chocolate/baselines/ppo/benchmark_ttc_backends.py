from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BenchWorldNumpy:
    centers: np.ndarray  # (A, C, 2)
    radii: np.ndarray  # (A, C)
    vel: np.ndarray  # (A, 2)
    fwd: np.ndarray  # (A, 2)
    road_points: np.ndarray  # (P, 2)


@dataclass
class BenchWorldTorch:
    centers: "torch.Tensor"  # (A, C, 2)
    radii: "torch.Tensor"  # (A, C)
    vel: "torch.Tensor"  # (A, 2)
    fwd: "torch.Tensor"  # (A, 2)
    road_points: "torch.Tensor"  # (P, 2)
    owner_idx: "torch.Tensor"  # (A*C,)


def _build_three_circles(
    base_xy: np.ndarray, yaw: np.ndarray, length: np.ndarray, width: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # radius = W/2, centers along spine at +/- (L/2 - r), 0
    radius = 0.5 * width
    spine_offset = np.maximum(0.0, 0.5 * length - radius)
    offsets = np.stack([-spine_offset, np.zeros_like(spine_offset), spine_offset], axis=1)  # (A,3)
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1).astype(np.float32)  # (A,2)
    cx = base_xy[:, 0:1] + offsets * fwd[:, 0:1]
    cy = base_xy[:, 1:2] + offsets * fwd[:, 1:2]
    centers = np.stack([cx, cy], axis=-1).astype(np.float32)  # (A,3,2)
    radii = np.repeat(radius[:, None], 3, axis=1).astype(np.float32)  # (A,3)
    return centers, radii


def make_numpy_worlds(
    *,
    num_worlds: int,
    agents_per_world: int,
    road_points_per_world: int,
    seed: int,
    extent_m: float,
    speed_mps: float,
) -> List[BenchWorldNumpy]:
    rng = np.random.default_rng(int(seed))
    worlds: List[BenchWorldNumpy] = []
    for _ in range(int(num_worlds)):
        A = int(agents_per_world)
        P = int(road_points_per_world)
        base_xy = rng.uniform(-extent_m, extent_m, size=(A, 2)).astype(np.float32)
        yaw = rng.uniform(-math.pi, math.pi, size=(A,)).astype(np.float32)
        length = rng.uniform(3.6, 5.8, size=(A,)).astype(np.float32)
        width = rng.uniform(1.7, 2.4, size=(A,)).astype(np.float32)
        centers, radii = _build_three_circles(base_xy, yaw, length, width)
        vel = rng.normal(0.0, speed_mps, size=(A, 2)).astype(np.float32)
        vnorm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel / np.maximum(vnorm, 1e-6) * rng.uniform(0.0, speed_mps, size=(A, 1)).astype(np.float32)
        fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1).astype(np.float32)
        fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)
        road_points = rng.uniform(-extent_m, extent_m, size=(P, 2)).astype(np.float32)
        worlds.append(
            BenchWorldNumpy(
                centers=centers,
                radii=radii,
                vel=vel,
                fwd=fwd,
                road_points=road_points,
            )
        )
    return worlds


def _vehicle_ttc_penalty_numpy(
    world: BenchWorldNumpy,
    *,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
) -> np.ndarray:
    centers = world.centers  # (A,C,2)
    radii = world.radii  # (A,C)
    vel = world.vel  # (A,2)
    fwd = world.fwd  # (A,2)
    A, C, _ = centers.shape
    penalties = np.zeros((A,), dtype=np.float32)
    centers_flat = centers.reshape(A * C, 2)
    radii_flat = radii.reshape(A * C)
    vel_flat = np.repeat(vel, C, axis=0)
    owner = np.repeat(np.arange(A, dtype=np.int32), C)
    inf = np.float32(np.inf)

    for ego in range(A):
        ego_centers = centers[ego]  # (C,2)
        ego_radii = radii[ego]  # (C,)
        ego_vel = vel[ego]  # (2,)
        ego_fwd = fwd[ego]  # (2,)
        mask = owner != ego
        other_centers = centers_flat[mask]  # (O,2)
        other_radii = radii_flat[mask]  # (O,)
        other_vel = vel_flat[mask]  # (O,2)

        rel = other_centers[None, :, :] - ego_centers[:, None, :]  # (C,O,2)
        rv = other_vel[None, :, :] - ego_vel[None, None, :]  # (C,O,2)
        rx = rel[:, :, 0]
        ry = rel[:, :, 1]
        rvx = rv[:, :, 0]
        rvy = rv[:, :, 1]
        r2 = rx * rx + ry * ry
        v2 = rvx * rvx + rvy * rvy
        rdotv = rx * rvx + ry * rvy
        combined_r = ego_radii[:, None] + other_radii[None, :]
        combined_r2 = combined_r * combined_r

        forward_dot = rx * ego_fwd[0] + ry * ego_fwd[1]
        forward_mask = forward_dot > 0.0
        ttc = np.full(r2.shape, inf, dtype=np.float32)
        overlap_mask = forward_mask & (r2 <= combined_r2)
        ttc[overlap_mask] = 0.0

        moving_mask = forward_mask & (~overlap_mask) & (v2 > 1e-6)
        if np.any(moving_mask):
            a = np.where(moving_mask, v2, 1.0)
            b = np.where(moving_mask, 2.0 * rdotv, 0.0)
            c = np.where(moving_mask, r2 - combined_r2, 0.0)
            disc = b * b - 4.0 * a * c
            valid_quad = moving_mask & (disc >= 0.0)
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            denom = 2.0 * np.maximum(a, 1e-6)
            t_enter = (-b - sqrt_disc) / denom
            t_exit = (-b + sqrt_disc) / denom
            valid_enter = valid_quad & (t_exit >= 0.0)
            ttc_quad = np.where(valid_enter, np.maximum(0.0, t_enter), inf).astype(np.float32)
            ttc = np.minimum(ttc, ttc_quad)

            unresolved = moving_mask & (~np.isfinite(ttc))
            if np.any(unresolved):
                dist = np.sqrt(np.maximum(r2, 1e-9))
                closing_speed = -rdotv / np.maximum(dist, 1e-6)
                clearance = np.maximum(0.0, dist - combined_r)
                valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1e-6)
                ttc_fb = np.where(
                    valid_fb,
                    clearance / np.maximum(closing_speed, 1e-6),
                    inf,
                ).astype(np.float32)
                ttc = np.minimum(ttc, ttc_fb)

        finite = ttc[np.isfinite(ttc)]
        if finite.size == 0:
            continue
        min_ttc = float(np.min(finite))
        if min_ttc < hard_ttc_s:
            abs_pen = max_pen
        else:
            abs_pen = min(max_pen, alpha / max(min_ttc, min_ttc_floor))
        penalties[ego] = -float(abs_pen)

    return penalties


def _road_ttc_penalty_numpy(
    world: BenchWorldNumpy,
    *,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
    radius_m: float,
) -> np.ndarray:
    centers = world.centers
    radii = world.radii
    vel = world.vel
    fwd = world.fwd
    points = world.road_points  # (P,2)
    A = centers.shape[0]
    penalties = np.zeros((A,), dtype=np.float32)
    radius2 = float(radius_m * radius_m)
    inf = np.float32(np.inf)

    for ego in range(A):
        ego_centers = centers[ego]  # (C,2)
        ego_r = radii[ego]  # (C,)
        ego_vel = vel[ego]  # (2,)
        ego_fwd = fwd[ego]  # (2,)

        rel = points[None, :, :] - ego_centers[:, None, :]  # (C,P,2)
        rx = rel[:, :, 0]
        ry = rel[:, :, 1]
        dist2 = rx * rx + ry * ry
        near_mask = dist2 <= radius2
        if not np.any(near_mask):
            continue
        forward_dot = rx * ego_fwd[0] + ry * ego_fwd[1]
        candidate_mask = near_mask & (forward_dot > 0.0)
        if not np.any(candidate_mask):
            continue

        dist = np.sqrt(np.maximum(dist2, 1e-12))
        ttc = np.full(dist.shape, inf, dtype=np.float32)
        overlap = candidate_mask & (dist <= ego_r[:, None])
        ttc[overlap] = 0.0

        dirs = rel / np.maximum(dist[:, :, None], 1e-6)
        closing_speed = dirs[:, :, 0] * ego_vel[0] + dirs[:, :, 1] * ego_vel[1]
        valid = candidate_mask & (closing_speed > 1e-6)
        if np.any(valid):
            clearance = np.maximum(0.0, dist - ego_r[:, None])
            ttc_vals = np.where(
                valid,
                clearance / np.maximum(closing_speed, 1e-6),
                inf,
            ).astype(np.float32)
            ttc = np.minimum(ttc, ttc_vals)

        finite = ttc[np.isfinite(ttc)]
        if finite.size == 0:
            continue
        min_ttc = float(np.min(finite))
        if min_ttc < hard_ttc_s:
            abs_pen = max_pen
        else:
            abs_pen = min(max_pen, alpha / max(min_ttc, min_ttc_floor))
        penalties[ego] = -float(abs_pen)

    return penalties


def _to_torch_world(world: BenchWorldNumpy, *, torch, device) -> BenchWorldTorch:
    A, C, _ = world.centers.shape
    owner_idx = torch.arange(A, device=device, dtype=torch.int64).repeat_interleave(C)
    return BenchWorldTorch(
        centers=torch.as_tensor(world.centers, dtype=torch.float32, device=device),
        radii=torch.as_tensor(world.radii, dtype=torch.float32, device=device),
        vel=torch.as_tensor(world.vel, dtype=torch.float32, device=device),
        fwd=torch.as_tensor(world.fwd, dtype=torch.float32, device=device),
        road_points=torch.as_tensor(world.road_points, dtype=torch.float32, device=device),
        owner_idx=owner_idx,
    )


def _vehicle_ttc_penalty_torch(
    world: BenchWorldTorch,
    *,
    torch,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
) -> "torch.Tensor":
    centers = world.centers
    radii = world.radii
    vel = world.vel
    fwd = world.fwd
    A, C, _ = centers.shape
    penalties = torch.zeros((A,), dtype=torch.float32, device=centers.device)
    centers_flat = centers.view(A * C, 2)
    radii_flat = radii.view(A * C)
    vel_flat = vel.repeat_interleave(C, dim=0)
    owner = world.owner_idx
    inf = torch.full((1,), float("inf"), dtype=torch.float32, device=centers.device)[0]

    for ego in range(A):
        ego_centers = centers[ego]  # (C,2)
        ego_radii = radii[ego]  # (C,)
        ego_vel = vel[ego]  # (2,)
        ego_fwd = fwd[ego]  # (2,)
        mask = owner != ego
        other_centers = centers_flat[mask]  # (O,2)
        other_radii = radii_flat[mask]  # (O,)
        other_vel = vel_flat[mask]  # (O,2)

        rel = other_centers.unsqueeze(0) - ego_centers.unsqueeze(1)
        rv = other_vel.unsqueeze(0) - ego_vel.view(1, 1, 2)
        rx = rel[:, :, 0]
        ry = rel[:, :, 1]
        rvx = rv[:, :, 0]
        rvy = rv[:, :, 1]
        r2 = rx * rx + ry * ry
        v2 = rvx * rvx + rvy * rvy
        rdotv = rx * rvx + ry * rvy
        combined_r = ego_radii.view(-1, 1) + other_radii.view(1, -1)
        combined_r2 = combined_r * combined_r

        forward_dot = rx * ego_fwd[0] + ry * ego_fwd[1]
        forward_mask = forward_dot > 0.0
        ttc = torch.full_like(r2, inf)
        overlap_mask = forward_mask & (r2 <= combined_r2)
        ttc = torch.where(overlap_mask, torch.zeros_like(ttc), ttc)

        moving_mask = forward_mask & (~overlap_mask) & (v2 > 1e-6)
        if bool(torch.any(moving_mask).item()):
            a = torch.where(moving_mask, v2, torch.ones_like(v2))
            b = torch.where(moving_mask, 2.0 * rdotv, torch.zeros_like(rdotv))
            c = torch.where(moving_mask, r2 - combined_r2, torch.zeros_like(r2))
            disc = b * b - 4.0 * a * c
            valid_quad = moving_mask & (disc >= 0.0)
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            denom = 2.0 * torch.clamp(a, min=1e-6)
            t_enter = (-b - sqrt_disc) / denom
            t_exit = (-b + sqrt_disc) / denom
            valid_enter = valid_quad & (t_exit >= 0.0)
            ttc_quad = torch.where(valid_enter, torch.clamp(t_enter, min=0.0), torch.full_like(ttc, inf))
            ttc = torch.minimum(ttc, ttc_quad)

            unresolved = moving_mask & (~torch.isfinite(ttc))
            if bool(torch.any(unresolved).item()):
                dist = torch.sqrt(torch.clamp(r2, min=1e-9))
                closing_speed = -rdotv / torch.clamp(dist, min=1e-6)
                clearance = torch.clamp(dist - combined_r, min=0.0)
                valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1e-6)
                ttc_fb = torch.where(
                    valid_fb,
                    clearance / torch.clamp(closing_speed, min=1e-6),
                    torch.full_like(ttc, inf),
                )
                ttc = torch.minimum(ttc, ttc_fb)

        finite = ttc[torch.isfinite(ttc)]
        if finite.numel() == 0:
            continue
        min_ttc = float(torch.min(finite).item())
        if min_ttc < hard_ttc_s:
            abs_pen = max_pen
        else:
            abs_pen = min(max_pen, alpha / max(min_ttc, min_ttc_floor))
        penalties[ego] = -float(abs_pen)

    return penalties


def _road_ttc_penalty_torch(
    world: BenchWorldTorch,
    *,
    torch,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
    radius_m: float,
) -> "torch.Tensor":
    centers = world.centers
    radii = world.radii
    vel = world.vel
    fwd = world.fwd
    points = world.road_points
    A = centers.shape[0]
    penalties = torch.zeros((A,), dtype=torch.float32, device=centers.device)
    radius2 = float(radius_m * radius_m)
    inf = torch.full((1,), float("inf"), dtype=torch.float32, device=centers.device)[0]

    for ego in range(A):
        ego_centers = centers[ego]  # (C,2)
        ego_r = radii[ego]  # (C,)
        ego_vel = vel[ego]  # (2,)
        ego_fwd = fwd[ego]  # (2,)

        rel = points.unsqueeze(0) - ego_centers.unsqueeze(1)  # (C,P,2)
        rx = rel[:, :, 0]
        ry = rel[:, :, 1]
        dist2 = rx * rx + ry * ry
        near_mask = dist2 <= radius2
        if not bool(torch.any(near_mask).item()):
            continue
        forward_dot = rx * ego_fwd[0] + ry * ego_fwd[1]
        candidate_mask = near_mask & (forward_dot > 0.0)
        if not bool(torch.any(candidate_mask).item()):
            continue

        dist = torch.sqrt(torch.clamp(dist2, min=1e-12))
        ttc = torch.full_like(dist, inf)
        overlap = candidate_mask & (dist <= ego_r.view(-1, 1))
        ttc = torch.where(overlap, torch.zeros_like(ttc), ttc)

        dirs = rel / torch.clamp(dist.unsqueeze(-1), min=1e-6)
        closing_speed = dirs[:, :, 0] * ego_vel[0] + dirs[:, :, 1] * ego_vel[1]
        valid = candidate_mask & (closing_speed > 1e-6)
        if bool(torch.any(valid).item()):
            clearance = torch.clamp(dist - ego_r.view(-1, 1), min=0.0)
            ttc_vals = torch.where(
                valid,
                clearance / torch.clamp(closing_speed, min=1e-6),
                torch.full_like(ttc, inf),
            )
            ttc = torch.minimum(ttc, ttc_vals)

        finite = ttc[torch.isfinite(ttc)]
        if finite.numel() == 0:
            continue
        min_ttc = float(torch.min(finite).item())
        if min_ttc < hard_ttc_s:
            abs_pen = max_pen
        else:
            abs_pen = min(max_pen, alpha / max(min_ttc, min_ttc_floor))
        penalties[ego] = -float(abs_pen)

    return penalties


def _time_numpy(
    worlds: List[BenchWorldNumpy],
    *,
    warmup: int,
    iters: int,
    bench_vehicle: bool,
    bench_road: bool,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
    road_radius_m: float,
) -> Dict[str, float]:
    checksum = 0.0
    for _ in range(int(warmup)):
        for w in worlds:
            if bench_vehicle:
                checksum += float(
                    np.sum(
                        _vehicle_ttc_penalty_numpy(
                            w,
                            alpha=alpha,
                            max_pen=max_pen,
                            min_ttc_floor=min_ttc_floor,
                            hard_ttc_s=hard_ttc_s,
                        )
                    )
                )
            if bench_road:
                checksum += float(
                    np.sum(
                        _road_ttc_penalty_numpy(
                            w,
                            alpha=alpha,
                            max_pen=max_pen,
                            min_ttc_floor=min_ttc_floor,
                            hard_ttc_s=hard_ttc_s,
                            radius_m=road_radius_m,
                        )
                    )
                )
    t0 = time.perf_counter()
    for _ in range(int(iters)):
        for w in worlds:
            if bench_vehicle:
                checksum += float(
                    np.sum(
                        _vehicle_ttc_penalty_numpy(
                            w,
                            alpha=alpha,
                            max_pen=max_pen,
                            min_ttc_floor=min_ttc_floor,
                            hard_ttc_s=hard_ttc_s,
                        )
                    )
                )
            if bench_road:
                checksum += float(
                    np.sum(
                        _road_ttc_penalty_numpy(
                            w,
                            alpha=alpha,
                            max_pen=max_pen,
                            min_ttc_floor=min_ttc_floor,
                            hard_ttc_s=hard_ttc_s,
                            radius_m=road_radius_m,
                        )
                    )
                )
    t1 = time.perf_counter()
    elapsed = max(t1 - t0, 1e-9)
    return {"elapsed_s": elapsed, "checksum": checksum}


def _time_torch_cuda(
    worlds_np: List[BenchWorldNumpy],
    *,
    device: str,
    warmup: int,
    iters: int,
    bench_vehicle: bool,
    bench_road: bool,
    alpha: float,
    max_pen: float,
    min_ttc_floor: float,
    hard_ttc_s: float,
    road_radius_m: float,
) -> Optional[Dict[str, float]]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None

    torch_device = torch.device(device)
    worlds = [_to_torch_world(w, torch=torch, device=torch_device) for w in worlds_np]

    checksum = torch.zeros((), dtype=torch.float32, device=torch_device)
    for _ in range(int(warmup)):
        for w in worlds:
            if bench_vehicle:
                checksum = checksum + torch.sum(
                    _vehicle_ttc_penalty_torch(
                        w,
                        torch=torch,
                        alpha=alpha,
                        max_pen=max_pen,
                        min_ttc_floor=min_ttc_floor,
                        hard_ttc_s=hard_ttc_s,
                    )
                )
            if bench_road:
                checksum = checksum + torch.sum(
                    _road_ttc_penalty_torch(
                        w,
                        torch=torch,
                        alpha=alpha,
                        max_pen=max_pen,
                        min_ttc_floor=min_ttc_floor,
                        hard_ttc_s=hard_ttc_s,
                        radius_m=road_radius_m,
                    )
                )
    torch.cuda.synchronize(torch_device)

    t0 = time.perf_counter()
    for _ in range(int(iters)):
        for w in worlds:
            if bench_vehicle:
                checksum = checksum + torch.sum(
                    _vehicle_ttc_penalty_torch(
                        w,
                        torch=torch,
                        alpha=alpha,
                        max_pen=max_pen,
                        min_ttc_floor=min_ttc_floor,
                        hard_ttc_s=hard_ttc_s,
                    )
                )
            if bench_road:
                checksum = checksum + torch.sum(
                    _road_ttc_penalty_torch(
                        w,
                        torch=torch,
                        alpha=alpha,
                        max_pen=max_pen,
                        min_ttc_floor=min_ttc_floor,
                        hard_ttc_s=hard_ttc_s,
                        radius_m=road_radius_m,
                    )
                )
    torch.cuda.synchronize(torch_device)
    t1 = time.perf_counter()
    elapsed = max(t1 - t0, 1e-9)
    return {"elapsed_s": elapsed, "checksum": float(checksum.item())}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark TTC kernel speed: NumPy (CPU) vs Torch CUDA."
    )
    p.add_argument("--num-worlds", type=int, default=16)
    p.add_argument("--agents-per-world", type=int, default=64)
    p.add_argument("--road-points-per-world", type=int, default=1500)
    p.add_argument("--extent-m", type=float, default=80.0)
    p.add_argument("--speed-mps", type=float, default=12.0)
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--bench-vehicle", action="store_true", default=True)
    p.add_argument("--bench-road", action="store_true", default=True)
    p.add_argument("--vehicle-only", action="store_true")
    p.add_argument("--road-only", action="store_true")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--max-pen", type=float, default=0.5)
    p.add_argument("--min-ttc-floor", type=float, default=0.5)
    p.add_argument("--hard-ttc-s", type=float, default=0.5)
    p.add_argument("--road-radius-m", type=float, default=40.0)
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bench_vehicle = bool(args.bench_vehicle)
    bench_road = bool(args.bench_road)
    if args.vehicle_only:
        bench_vehicle, bench_road = True, False
    if args.road_only:
        bench_vehicle, bench_road = False, True
    if not bench_vehicle and not bench_road:
        raise ValueError("Nothing to benchmark: enable vehicle and/or road TTC.")

    worlds = make_numpy_worlds(
        num_worlds=args.num_worlds,
        agents_per_world=args.agents_per_world,
        road_points_per_world=args.road_points_per_world,
        seed=args.seed,
        extent_m=args.extent_m,
        speed_mps=args.speed_mps,
    )

    workload_agents = int(args.num_worlds) * int(args.agents_per_world)
    workload_iters = int(args.iters)
    bench_label = (
        "vehicle+road"
        if (bench_vehicle and bench_road)
        else ("vehicle" if bench_vehicle else "road")
    )
    print(
        "[ttc-bench] setup "
        f"mode={bench_label} worlds={args.num_worlds} agents_per_world={args.agents_per_world} "
        f"road_points_per_world={args.road_points_per_world} iters={args.iters}"
    )

    np_res = _time_numpy(
        worlds,
        warmup=args.warmup_iters,
        iters=args.iters,
        bench_vehicle=bench_vehicle,
        bench_road=bench_road,
        alpha=args.alpha,
        max_pen=args.max_pen,
        min_ttc_floor=args.min_ttc_floor,
        hard_ttc_s=args.hard_ttc_s,
        road_radius_m=args.road_radius_m,
    )
    np_iter_ms = 1000.0 * np_res["elapsed_s"] / max(workload_iters, 1)
    np_agents_per_s = (workload_agents * workload_iters) / max(np_res["elapsed_s"], 1e-9)
    print(
        "[ttc-bench] numpy "
        f"avg_iter_ms={np_iter_ms:.3f} "
        f"agents_per_s={np_agents_per_s:,.1f} checksum={np_res['checksum']:.3f}"
    )

    cuda_res = _time_torch_cuda(
        worlds,
        device=args.device,
        warmup=args.warmup_iters,
        iters=args.iters,
        bench_vehicle=bench_vehicle,
        bench_road=bench_road,
        alpha=args.alpha,
        max_pen=args.max_pen,
        min_ttc_floor=args.min_ttc_floor,
        hard_ttc_s=args.hard_ttc_s,
        road_radius_m=args.road_radius_m,
    )

    out: Dict[str, object] = {
        "setup": {
            "mode": bench_label,
            "num_worlds": int(args.num_worlds),
            "agents_per_world": int(args.agents_per_world),
            "road_points_per_world": int(args.road_points_per_world),
            "iters": int(args.iters),
            "warmup_iters": int(args.warmup_iters),
            "device": str(args.device),
        },
        "numpy": {
            "elapsed_s": float(np_res["elapsed_s"]),
            "avg_iter_ms": float(np_iter_ms),
            "agents_per_s": float(np_agents_per_s),
            "checksum": float(np_res["checksum"]),
        },
    }

    if cuda_res is None:
        print("[ttc-bench] torch_cuda unavailable (torch or CUDA not ready in this env).")
        out["torch_cuda"] = None
    else:
        cu_iter_ms = 1000.0 * cuda_res["elapsed_s"] / max(workload_iters, 1)
        cu_agents_per_s = (workload_agents * workload_iters) / max(cuda_res["elapsed_s"], 1e-9)
        speedup = np_res["elapsed_s"] / max(cuda_res["elapsed_s"], 1e-9)
        print(
            "[ttc-bench] torch_cuda "
            f"device={args.device} avg_iter_ms={cu_iter_ms:.3f} "
            f"agents_per_s={cu_agents_per_s:,.1f} checksum={cuda_res['checksum']:.3f}"
        )
        print(f"[ttc-bench] speedup torch_cuda_vs_numpy={speedup:.3f}x")
        out["torch_cuda"] = {
            "elapsed_s": float(cuda_res["elapsed_s"]),
            "avg_iter_ms": float(cu_iter_ms),
            "agents_per_s": float(cu_agents_per_s),
            "checksum": float(cuda_res["checksum"]),
            "speedup_vs_numpy": float(speedup),
        }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[ttc-bench] wrote json: {out_path}")


if __name__ == "__main__":
    main()
