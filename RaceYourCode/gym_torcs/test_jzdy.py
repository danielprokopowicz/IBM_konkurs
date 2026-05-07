"""
TORCS RL v2 - Szybsza jazda przez curriculum prędkości
=======================================================
Główna zmiana: target_speed rośnie z czasem (curriculum learning)
- ep 0-50:   target=50 km/h  (bezpieczna nauka toru)
- ep 50-150: target=80 km/h
- ep 150+:   target=120 km/h (agresywna jazda)

Dodatkowo: nagroda za prędkość jest skalowana przez docelową prędkość
żeby agent uczył się JECHAĆ SZYBKO a nie tylko nie wypadać.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import os
import traceback
import sys

PI = 3.14159265359


def get_state(env):
    try:
        d = env.client.S.d
        track = d.get('track', [100.0]*19)
        return np.array([
            float(d.get('angle',    0.0)) / PI,
            float(d.get('trackPos', 0.0)),
            float(d.get('speedX',   0.0)) / 300.0,
            float(track[4])  / 200.0,
            float(track[9])  / 200.0,
            float(track[14]) / 200.0,
        ], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def get_raw(env):
    try:
        d = env.client.S.d
        return (
            float(d.get('angle',      0.0)),
            float(d.get('trackPos',   0.0)),
            float(d.get('speedX',     0.0)),
            float(d.get('distRaced',  0.0)),
            float(d.get('lastLapTime',0.0)),
            float(d.get('curLapTime', 0.0)),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def load_best(actor, critic=None):
    for fname in ["snakeoil_fastest.pth", "snakeoil_stable.pth", "bc_final.pth"]:
        if os.path.exists(fname):
            if fname == "bc_final.pth":
                actor.load_state_dict(torch.load(fname))
            else:
                ck = torch.load(fname)
                actor.load_state_dict(ck['actor'])
                if critic and 'critic' in ck:
                    critic.load_state_dict(ck['critic'])
            print(f"✅ Wczytano: {fname}", flush=True)
            return actor, critic, fname
    return actor, critic, None


class Actor(nn.Module):
    def __init__(self, state_dim=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
        )
        self.steer_head = nn.Linear(64, 1)
        self.accel_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.fc(x)
        return torch.cat([
            torch.tanh(self.steer_head(h)),
            torch.sigmoid(self.accel_head(h)),
        ], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim=6, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),                     nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        return self.fc(torch.cat([s, a], dim=1))


# ============================================================
# CURRICULUM PRĘDKOŚCI
# Agent uczy się toru wolno, potem stopniowo przyspiesza
# ============================================================
def get_target_speed(ep, lap_count):
    """
    Prędkość docelowa rośnie z czasem i z liczbą ukończonych okrążeń.
    Ukończone okrążenia = dowód że agent zna tor = można jechać szybciej.
    """
    # Baza: rośnie z episodami
    speed_from_ep = min(120.0, 50.0 + ep * 0.5)

    # Bonus za ukończone okrążenia: każde +5 km/h, max +40
    speed_from_laps = min(40.0, lap_count * 5.0)

    return speed_from_ep + speed_from_laps


def compute_reward(angle, tpos, speed, target_speed, dist_ep, done_lap, done_crash):
    """
    Nagroda uwzględniająca aktualną prędkość docelową.
    Im wyższy target_speed, tym bardziej nagradzamy za jazdę szybką.
    """
    if done_crash:
        return -50.0

    if done_lap:
        return 100.0

    # Postęp: nagroda rośnie gdy jedzie szybko WZDŁUŻ toru
    progress = speed * np.cos(angle) / target_speed  # ~[0,1] przy idealnej jeździe

    # Kara za pozycję: quadratyczna, rośnie przy krawędzi
    edge_pen = (tpos ** 2) * 1.5

    # Mała kara za kąt (nie jedź bokiem)
    angle_pen = abs(np.sin(angle)) * 0.3

    return float(np.clip(progress - edge_pen - angle_pen, -5.0, 2.0))


# ============================================================
# REPLAY
# ============================================================
def replay_mode(lap_threshold=2200):
    print("\n" + "="*60, flush=True)
    print("  TRYB REPLAY", flush=True)
    print("="*60, flush=True)

    from torcs_env import TorcsEnv
    env   = TorcsEnv(vision=False, throttle=True, gear_change=False)
    actor = Actor(6)
    actor, _, fname = load_best(actor)

    if fname is None:
        print("❌ Brak modelu!", flush=True)
        env.end()
        return

    actor.eval()

    for run in range(3):
        env.reset()
        for _ in range(5):
            try:
                angle_r = float(env.client.S.d.get('angle', 0.0))
                s_f = float(np.clip(angle_r * 15.0 / PI, -1, 1))
                env.step(np.array([s_f, 0.3, 0.0], dtype=np.float64))
            except Exception:
                pass

        state = get_state(env)
        _, _, _, dist_start, _, _ = get_raw(env)
        done  = False
        steps = 0

        print(f"\n  --- Przejazd #{run+1} ---", flush=True)

        while not done:
            with torch.no_grad():
                a = actor(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]

            steer = float(np.clip(a[0], -1.0, 1.0))
            accel = float(np.clip(a[1],  0.0, 1.0))

            try:
                _, _, done, _ = env.step(np.array([steer, accel, 0.0], dtype=np.float64))
            except Exception:
                break

            next_state = get_state(env)
            angle, tpos, speed, dist_raced, last_lap, cur_lap = get_raw(env)
            dist_ep = dist_raced - dist_start

            if steps % 50 == 0:
                print(f"    {steps:5d} | {dist_ep:6.0f}m | {speed:5.1f}km/h | "
                      f"angle={angle:+.3f} | tpos={tpos:+.4f} | t={cur_lap:.1f}s", flush=True)

            state = next_state
            steps += 1

            if dist_ep > lap_threshold:
                print(f"\n  🏁 Okrążenie! Czas={last_lap:.2f}s Kroki={steps}", flush=True)
                done = True
            if abs(tpos) > 0.99:
                print(f"  ❌ Wypadł na kroku {steps}", flush=True)
                done = True
            if steps >= 10000:
                done = True

    env.end()


# ============================================================
# FAZA RL z curriculum prędkości
# ============================================================
def rl_phase(actor, critic, env, episodes=1000, lap_threshold=2200):
    print("\n" + "="*60, flush=True)
    print("  FAZA RL z curriculum prędkości", flush=True)
    print(f"  Próg okrążenia: {lap_threshold}m", flush=True)
    print("="*60, flush=True)

    t_actor  = copy.deepcopy(actor)
    t_critic = copy.deepcopy(critic)
    a_opt = optim.Adam(actor.parameters(),  lr=0.00005)
    c_opt = optim.Adam(critic.parameters(), lr=0.001)
    memory = deque(maxlen=100000)

    best_dist     = 0.0
    best_lap_time = 9999.0
    bad           = 0
    gamma         = 0.99
    tau           = 0.005
    lap_count     = 0

    for ep in range(episodes):
        env.reset()

        # Curriculum: prędkość docelowa rośnie z czasem
        # Nagradzamy za jazde z dowolna predkoscia - im szybciej tym lepiej
        target_speed = 120.0
        state = get_state(env)
        _, _, _, dist_start, _, _ = get_raw(env)
        done    = False
        steps   = 0
        total_r = 0.0

        # Szum: większy na początku, maleje gdy agent jest już dobry
        noise = max(0.01, 0.15 - ep * 0.0001)

        while not done:
            try:
                s_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    a_raw = actor(s_t).numpy()[0]

                steer = float(np.clip(a_raw[0] + np.random.randn() * noise,      -1.0, 1.0))
                accel = float(np.clip(a_raw[1] + np.random.randn() * noise * 0.3, 0.0, 1.0))

                _, _, done, _ = env.step(np.array([steer, accel, 0.0], dtype=np.float64))
                next_state = get_state(env)

                angle, tpos, speed, dist_raced, last_lap, cur_lap = get_raw(env)
                dist_ep = dist_raced - dist_start

                done_lap   = dist_ep > lap_threshold
                done_crash = abs(tpos) > 0.99

                reward = compute_reward(angle, tpos, speed, target_speed,
                                        dist_ep, done_lap, done_crash)

                if done_lap:
                    done = True
                    lap_count += 1
                    if last_lap > 0 and last_lap < best_lap_time:
                        best_lap_time = last_lap
                        torch.save(
                            {'actor': actor.state_dict(), 'critic': critic.state_dict()},
                            "snakeoil_fastest.pth"
                        )
                        print(f"  🔥 NOWY REKORD: {best_lap_time:.2f}s! "
                              f"(target_speed={target_speed:.0f})", flush=True)
                    else:
                        print(f"  🏁 LAP #{lap_count} {last_lap:.2f}s "
                              f"(rekord={best_lap_time:.2f}s "
                              f"target={target_speed:.0f}km/h)", flush=True)

                if done_crash:
                    done = True

                memory.append((
                    state,
                    np.array([steer, accel], dtype=np.float32),
                    reward,
                    next_state,
                    float(done)
                ))
                state   = next_state
                total_r += reward
                steps   += 1

                if steps >= 10000:
                    done = True

            except Exception as e:
                print(f"  Błąd: {e}", flush=True)
                break

        _, _, speed_end, dist_raced, last_lap, _ = get_raw(env)
        dist_ep = dist_raced - dist_start

        print(f"  ep {ep:4d}: {steps:5d}kr | dist={dist_ep:5.0f}m | "
              f"r={total_r:7.1f} | target={target_speed:.0f}km/h | "
              f"laps={lap_count} | szum={noise:.3f}", flush=True)

        if dist_ep > best_dist:
            best_dist = dist_ep
            torch.save(
                {'actor': actor.state_dict(), 'critic': critic.state_dict()},
                "snakeoil_stable.pth"
            )
            print(f"  📍 REKORD DYSTANSU: {dist_ep:.0f}m", flush=True)
            bad = 0
        else:
            bad = bad + 1 if dist_ep < best_dist * 0.5 else 0

        if bad >= 15:
            actor, critic, _ = load_best(actor, critic)
            t_actor  = copy.deepcopy(actor)
            t_critic = copy.deepcopy(critic)
            bad = 0
            print("  🔄 Reset do najlepszego modelu", flush=True)

        if len(memory) < 1000:
            continue

        # Uczenie
        for _ in range(150):
            batch = random.sample(memory, 128)
            bs    = torch.FloatTensor(np.array([b[0] for b in batch]))
            ba    = torch.FloatTensor(np.array([b[1] for b in batch]))
            br    = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
            bns   = torch.FloatTensor(np.array([b[3] for b in batch]))
            bd    = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1)

            with torch.no_grad():
                na = t_actor(bns)
                tq = br + gamma * t_critic(bns, na) * (1 - bd)

            c_loss = nn.MSELoss()(critic(bs, ba), tq)
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()

            a_loss = -critic(bs, actor(bs)).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()

            for tp, p in zip(t_actor.parameters(), actor.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)
            for tp, p in zip(t_critic.parameters(), critic.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Próg okrążenia - zmień jeśli twój tor jest innej długości
    # Sprawdź w logach BC: wartość dist= przy ukończonym okrążeniu
    LAP_THRESHOLD = 2200  # metry

    if "--replay" in sys.argv:
        replay_mode(lap_threshold=LAP_THRESHOLD)
    else:
        from torcs_env import TorcsEnv
        env    = TorcsEnv(vision=False, throttle=True, gear_change=False)
        actor  = Actor(8)
        critic = Critic(8, 2)

        actor, critic, fname = load_best(actor, critic)
        if fname is None:
            print("❌ Brak modelu! Uruchom najpierw fazę BC.", flush=True)
            env.end()
            sys.exit(1)

        try:
            rl_phase(actor, critic, env,
                     episodes=1000,
                     lap_threshold=LAP_THRESHOLD)
        except KeyboardInterrupt:
            print("\nZatrzymano.", flush=True)
        except Exception as e:
            print(f"\nBłąd: {e}", flush=True)
            traceback.print_exc()
        finally:
            env.end()