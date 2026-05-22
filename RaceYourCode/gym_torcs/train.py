"""
train.py — AI Training Pipeline for TORCS Corkscrew Track
==========================================================
AI controls: steer, accel, brake (full control)
Code controls: gears only

Usage:
  python train.py collect     # Phase 1: collect data
  python train.py bc          # Phase 2: Behavioral Cloning
  python train.py play        # Run trained model
"""

import numpy as np
import os
import json

import sys
COMMAND = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = [sys.argv[0]]

import snakeoil3_gym as snakeoil3

PI = 3.14159265359

# ============================================================
# CONFIG
# ============================================================
STATE_DIM     = 14
ACTION_DIM    = 3       # steer, accel, brake
LAP_THRESHOLD = 3600
DATA_FILE     = "driving_data.json"
PORT          = 3001


# ============================================================
# STATE: 14-dim — unchanged
# ============================================================
def get_state(S):
    track = S.get('track', [100.0] * 19)
    return np.array([
        float(S.get('angle', 0.0)) / PI,
        float(S.get('trackPos', 0.0)),
        float(S.get('speedX', 0.0)) / 300.0,
        float(S.get('speedY', 0.0)) / 300.0,
        float(S.get('rpm', 0.0)) / 10000.0,
        float(track[0]) / 200.0,
        float(track[3]) / 200.0,
        float(track[5]) / 200.0,
        float(track[7]) / 200.0,
        float(track[9]) / 200.0,
        float(track[11]) / 200.0,
        float(track[13]) / 200.0,
        float(track[16]) / 200.0,
        float(track[18]) / 200.0,
    ], dtype=np.float32)


# ============================================================
# TARGET SPEED — unchanged
# ============================================================
def get_target_speed(track):
    front = [track[i] for i in range(7, 12)]
    front_min = min(front)
    wide_front = [track[i] for i in range(5, 14)]
    wide_min = min(wide_front)
    look_ahead = min(front_min, wide_min)

    if look_ahead > 180:   return 320
    elif look_ahead > 150: return 300
    elif look_ahead > 100: return 250
    elif look_ahead > 80:  return 220
    elif look_ahead > 50:  return 190
    elif look_ahead > 38:  return 160
    elif look_ahead > 30:  return 140
    elif look_ahead > 25:  return 130
    elif look_ahead > 18:  return 105
    elif look_ahead > 12:  return 85
    elif look_ahead > 8:   return 65
    else:                  return 45


def detect_corner_direction(track):
    left_sum  = sum(track[0:7])
    right_sum = sum(track[12:19])
    diff = right_sum - left_sum
    if abs(diff) < 30:
        return 0
    return 1 if diff > 0 else -1


# ============================================================
# BRAKE + GEARS — used only during collect, not in play
# ============================================================
def apply_brake_and_gears(S, R):
    speed_x    = float(S.get('speedX', 0))
    angle      = float(S.get('angle', 0))
    speed_y    = float(S.get('speedY', 0))
    track      = S.get('track', [100.0] * 19)
    wheel_spin = S.get('wheelSpinVel', [0, 0, 0, 0])

    target_speed = get_target_speed(track)
    front_min = min(track[7], track[8], track[9], track[10], track[11])

    if front_min < 50 and speed_x > 140:
        R['brake'] = 0.7; R['accel'] = 0.0
        _apply_gears(speed_x, R); return
    if front_min < 30 and speed_x > 100:
        R['brake'] = 0.9; R['accel'] = 0.0
        _apply_gears(speed_x, R); return

    if speed_x > target_speed:
        overspeed = speed_x - target_speed
        if overspeed > 80:
            R['brake'] = 1.0; R['accel'] = 0.0
        elif overspeed > 50:
            R['brake'] = 0.7; R['accel'] = 0.0
        elif overspeed > 25:
            R['brake'] = 0.4; R['accel'] = 0.0
        else:
            R['brake'] = 0.15
    else:
        R['brake'] = 0.0

    if abs(angle) > 0.5:
        R['brake'] = max(R.get('brake', 0), 0.7); R['accel'] = 0.0
    elif abs(angle) > 0.3:
        R['brake'] = max(R.get('brake', 0), 0.3)
        R['accel'] = min(R.get('accel', 0), 0.2)

    if abs(speed_y) > 30:
        R['brake'] = max(R.get('brake', 0), 0.2)
        R['accel'] = min(R.get('accel', 0), 0.3)

    if speed_x > 10:
        slip = sum(abs(wheel_spin[i] * 0.3 - speed_x) for i in range(4))
        if slip / 4 > speed_x * 0.5:
            R['brake'] = R.get('brake', 0) * 0.6

    if (wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1]) > 8:
        R['accel'] = max(0.0, R.get('accel', 0) - 0.15)

    _apply_gears(speed_x, R)


def _apply_gears(speed_x, R):
    gear = 1
    if speed_x > 40:  gear = 2
    if speed_x > 70:  gear = 3
    if speed_x > 100: gear = 4
    if speed_x > 135: gear = 5
    if speed_x > 170: gear = 6
    R['gear'] = gear


# ============================================================
# RULE-BASED DRIVER — steer + accel (unchanged)
# ============================================================
STEER_LOCK = 0.366

def rule_based_steer_accel(S):
    angle     = float(S.get('angle', 0.0))
    track_pos = float(S.get('trackPos', 0.0))
    speed_x   = float(S.get('speedX', 0.0))
    track     = S.get('track', [100.0] * 19)

    target_speed = get_target_speed(track)

    steer = angle * 0.8 / STEER_LOCK - track_pos * 0.35

    corner_dir = detect_corner_direction(track)
    if corner_dir != 0 and speed_x > 40 and abs(track_pos) < 0.4:
        steer -= corner_dir * 0.1

    steer = max(-1.0, min(1.0, steer))

    speed_diff = target_speed - speed_x
    if speed_x < target_speed:
        if speed_diff > 60:    accel = 1.0
        elif speed_diff > 30:  accel = 0.8
        elif speed_diff > 10:  accel = 0.5
        else:                  accel = 0.3
    else:
        accel = 0.0

    if speed_x < 10: accel = max(accel, 1.0)
    if abs(angle) > 0.7: accel = 0.0
    elif abs(angle) > 0.4: accel = min(accel, 0.2)

    if abs(track_pos) > 0.95:
        steer = -track_pos * 0.5; accel = 0.15
    if abs(track_pos) > 1.0:
        steer = -track_pos * 0.8; accel = 0.05

    look = min(min(track[7:12]), min(track[5:14]))
    if abs(track_pos) > 0.7 and look < 33 and accel > 0.2:
        accel = 0.15

    return np.array([steer, accel], dtype=np.float32)


# ============================================================
# NEW: rule_based_action_with_brake
# Returns [steer, accel, brake] — used only for data collection
# ============================================================
def rule_based_action_with_brake(S):
    """Return [steer, accel, brake] for BC training."""
    action_2 = rule_based_steer_accel(S)
    steer = float(action_2[0])
    accel = float(action_2[1])

    # Compute brake using apply_brake_and_gears logic
    R = {'steer': steer, 'accel': accel, 'brake': 0.0}
    apply_brake_and_gears(S, R)
    brake = float(R.get('brake', 0.0))
    # Sync accel with what apply_brake_and_gears decided
    accel = float(R.get('accel', accel))

    return np.array([steer, accel, brake], dtype=np.float32)


def rule_based_drive_full(S, R):
    """Full rule-based drive for collect — no print."""
    action = rule_based_action_with_brake(S)
    R['steer'] = float(action[0])
    R['accel'] = float(action[1])
    R['brake'] = float(action[2])
    _apply_gears(float(S.get('speedX', 0)), R)


# ============================================================
# PHASE 1: COLLECT DATA
# ============================================================
def collect_data(num_laps=50, max_steps=500000):
    print("\n" + "=" * 60)
    print("  PHASE 1: Collecting data (aggressive driver)")
    print("  Make sure TORCS is running with Corkscrew track!")
    print("=" * 60)

    
    C = snakeoil3.Client(p=PORT)
    C.MAX_STEPS = max_steps

    # Load existing data if available
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            all_data = json.load(f)
        print(f"  Loaded {len(all_data)} existing samples, continuing...")
    else:
        all_data = []


    lap_count = 0
    prev_last_lap = 0.0

    for step in range(max_steps, 0, -1):
        C.get_servers_input()
        S = C.S.d

        state = get_state(S)
        action = rule_based_action_with_brake(S)  # [steer, accel, brake]
        all_data.append({
            'state': state.tolist(),
            'action': action.tolist(),
        })

        rule_based_drive_full(S, C.R.d)

        dist  = float(S.get('distRaced', 0.0))
        speed = float(S.get('speedX', 0.0))
        cur_step = max_steps - step

        if cur_step % 500 == 0 and cur_step > 0:
            print(f"    step {cur_step:5d} | dist={dist:6.0f}m | "
                  f"speed={speed:5.1f}km/h | "
                  f"tpos={S.get('trackPos', 0):+.3f}")

        last_lap = float(S.get('lastLapTime', 0.0))
        if last_lap > 0 and last_lap != prev_last_lap:
            lap_count += 1
            prev_last_lap = last_lap
            print(f"\n  Lap {lap_count} completed! Time: {last_lap:.2f}s | "
                  f"Total dist: {dist:.0f}m")
            if lap_count >= num_laps:
                print(f"  Collected {num_laps} laps, stopping.")
                break

        C.respond_to_server()

    C.shutdown()

    with open(DATA_FILE, 'w') as f:
        json.dump(all_data, f)
    print(f"\n  Collected {len(all_data)} samples -> {DATA_FILE}")
    # Verify brake distribution
    import json as _json
    with open(DATA_FILE) as f:
        d = _json.load(f)
    brakes = [x['action'][2] for x in d]
    brake_nonzero = sum(1 for b in brakes if b > 0.05)
    print(f"  Brake>0.05 samples: {brake_nonzero} ({100*brake_nonzero/len(brakes):.1f}%)")


# ============================================================
# PHASE 2: BEHAVIORAL CLONING — unchanged except action_dim=3
# ============================================================
def train_bc(epochs=500, batch_size=256):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import Actor

    print("\n" + "=" * 60)
    print("  PHASE 2: Behavioral Cloning")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print(f"  No data! Run: python train.py collect")
        return

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")
    print(f"  Action dim: {len(data[0]['action'])} (steer, accel, brake)")

    states  = torch.FloatTensor([d['state']  for d in data])
    actions = torch.FloatTensor([d['action'] for d in data])

    actor     = Actor(STATE_DIM)
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    loss_fn   = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        perm      = torch.randperm(len(states))
        states_s  = states[perm]
        actions_s = actions[perm]
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(states_s), batch_size):
            batch_s = states_s[i:i + batch_size]
            batch_a = actions_s[i:i + batch_size]
            pred    = actor(batch_s)
            loss    = loss_fn(pred, batch_a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if epoch % 10 == 0 or avg_loss < best_loss:
            marker = ""
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(actor.state_dict(), "bc_model.pth")
                marker = " <- BEST saved"
            print(f"  epoch {epoch:4d} | loss={avg_loss:.6f}{marker}")

    print(f"\n  BC complete! Best loss={best_loss:.6f} -> bc_model.pth")



# ============================================================
# PLAY MODE — AI controls steer, accel, brake. Code: gears only.
# ============================================================
def play_model():
    import torch
    from model import Actor

    print("\n" + "=" * 60)
    print("  PLAY MODE — AI controls steer, accel, brake")
    print("=" * 60)

    actor = Actor(STATE_DIM)

    loaded = False
    for fname in ["bc_model.pth"]:
        if os.path.exists(fname):
            if fname == "bc_model.pth":
                actor.load_state_dict(
                    torch.load(fname, map_location='cpu', weights_only=False))
            else:
                ck = torch.load(fname, map_location='cpu', weights_only=False)
                actor.load_state_dict(ck['actor'])
            print(f"  Loaded: {fname}")
            loaded = True
            break

    if not loaded:
        print("  NO MODEL FOUND!")
        return

    actor.eval()

    C = snakeoil3.Client(p=PORT)
    C.MAX_STEPS = 50000
    C.get_servers_input()
    S = C.S.d

    dist_start = float(S.get('distRaced', 0))
    steps = 0

    print("  Driving...")

    while True:
        state = get_state(S)
        with torch.no_grad():
            a = actor(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]

        R = C.R.d
        R['steer'] = float(np.clip(a[0], -1, 1))
        R['accel'] = float(np.clip(a[1],  0, 1))
        R['brake'] = float(np.clip(a[2],  0, 1))

        # Gears only — AI controls everything else
        speed_x = float(S.get('speedX', 0))
        gear = 1
        if speed_x > 40:  gear = 2
        if speed_x > 70:  gear = 3
        if speed_x > 100: gear = 4
        if speed_x > 135: gear = 5
        if speed_x > 170: gear = 6
        R['gear'] = gear

        C.respond_to_server()
        C.get_servers_input()
        S = C.S.d

        dist_ep = float(S.get('distRaced', 0)) - dist_start
        steps  += 1

        if steps % 100 == 0:
            print(f"    {steps:5d} | {dist_ep:6.0f}m | "
                  f"{float(S.get('speedX', 0)):5.1f}km/h | "
                  f"lap={float(S.get('curLapTime', 0)):.1f}s")

        if dist_ep > LAP_THRESHOLD * 2:
            print(f"\n  Done! Distance: {dist_ep:.0f}m")
            break
        if steps > 30000:
            break

    C.shutdown()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if COMMAND == "collect":
        collect_data(num_laps=50)
    elif COMMAND == "bc":
        train_bc(epochs=500)
    elif COMMAND == "play":
        play_model()
    else:
        print("TORCS Corkscrew AI Training Pipeline")
        print("=" * 45)
        print("  python train.py collect   # Phase 1: collect data")
        print("  python train.py bc        # Phase 2: Behavioral Cloning")
        print("  python train.py play      # Run trained model")
        print()
        print("Order: collect -> bc -> play")