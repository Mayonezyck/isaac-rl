import numpy as np

def heuristic_policy(obs7: np.ndarray,
                     *,
                     k_steer: float = 1.8,
                     k_thr: float = 0.7,
                     k_brake: float = 0.4,
                     thr_clip=(0.0, 1.0),
                     steer_clip=(-1.0, 1.0),
                     brake_clip=(0.0, 1.0),
                     slow_down_dist_n: float = 0.01,
                     target_vx_n: float = 0.25) -> np.ndarray:
    """
    obs7: (N,7) = [relx_n, rely_n, sin_he, cos_he, dist_n, vx_n, vy_n]
    returns (N,3) = [thr, steer, brake]
    """
    relx_n = obs7[:, 0]
    rely_n = obs7[:, 1]     # not essential, but can be used lightly
    sin_he = obs7[:, 2]
    cos_he = obs7[:, 3]
    dist_n = obs7[:, 4]
    vx_n   = obs7[:, 5]
    #print('inside policy?')
    # -------------------------
    # Steering: use heading error, not lateral offset
    # -------------------------
    steer = k_steer * sin_he

    # Optional small “lateral” term to reduce side-slip / help converge
    # (keep it small so it doesn't cause orbiting)
    steer += 0.3 * rely_n

    steer = np.clip(steer, steer_clip[0], steer_clip[1])

    # -------------------------
    # Throttle: only push when facing the goal
    # -------------------------
    # facing_factor in [0,1] (0 if facing >90deg away)
    # --- Throttle: based on distance, not relx ---
    facing = np.clip(cos_he, 0.0, 1.0)

    # map dist_n -> [0,1] where 0 means "at goal", 1 means "far"
    slow = np.clip(dist_n / max(slow_down_dist_n, 1e-6), 0.0, 1.0)

    # throttle floor so it doesn't crawl while turning
    thr_min = 0.18  # try 0.12~0.25
    thr = (thr_min + (k_thr - thr_min) * slow) * (0.35 + 0.65 * facing)
    thr = np.clip(thr, thr_clip[0], thr_clip[1])


    # -------------------------
    # Brake: if facing away or too fast near goal
    # -------------------------
    # brake when cos_he < 0 (facing away)
    brake_turn = np.clip(-cos_he, 0.0, 1.0)

    # brake if moving faster than target near goal
    too_fast = np.clip(vx_n - target_vx_n, 0.0, 1.0)
    brake_slow = too_fast * np.clip((slow_down_dist_n - dist_n) / max(slow_down_dist_n, 1e-6), 0.0, 1.0)

    brake = k_brake * (0.7 * brake_turn + 0.3 * brake_slow)
    brake = np.clip(brake, brake_clip[0], brake_clip[1])

    U = np.stack([thr.astype(np.float32), steer.astype(np.float32), brake.astype(np.float32)], axis=1)
    return U
