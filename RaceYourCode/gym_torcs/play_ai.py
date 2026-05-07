import numpy as np
import torch
import torch.nn as nn
import os

PI = 3.14159265359

def get_state(env):
    try:
        d = env.client.S.d
        angle     = float(d.get('angle',    0.0))
        track_pos = float(d.get('trackPos', 0.0))
        speed_x   = float(d.get('speedX',   0.0))
        track     = d.get('track', [100.0]*19)
        return np.array([
            angle / PI,
            track_pos,
            speed_x / 300.0,
            float(track[4])  / 200.0,
            float(track[9])  / 200.0,
            float(track[14]) / 200.0,
        ], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)

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

def play():
    print("🏁 TRYB POKAZOWY: Rozgrzewka + Lotne Okrążenie!")
    from torcs_env import TorcsEnv
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    
    actor = Actor(state_dim=6)
    
    if os.path.exists("snakeoil_fastest.pth"):
        print("🏆 Wczytuję mózg: snakeoil_fastest.pth")
        checkpoint = torch.load("snakeoil_fastest.pth")
        actor.load_state_dict(checkpoint['actor'])
        actor.eval() 
    else:
        print("❌ Brak pliku!")
        return

    env.reset()
    done = False
    
    print("Okrążenie 1: Start zatrzymany (Może być niestabilnie!)")
    
    while not done:
        state = get_state(env)
        
        with torch.no_grad():
            # SUROWA DECYZJA AI - zero ingerencji
            action_raw = actor(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]
        
        steer = float(np.clip(action_raw[0], -1.0, 1.0))
        accel = float(np.clip(action_raw[1], 0.0, 1.0))
        
        try:
            _, _, done, _ = env.step(np.array([steer, accel, 0.0], dtype=np.float64))
        except Exception:
            break
            
        try:
            dist_raced = float(env.client.S.d.get('distRaced', 0.0))
        except Exception:
            dist_raced = 0.0
            
        # Pozwalamy na przejechanie około 3 okrążeń toru (6200 metrów), żeby pokazać pełnię możliwości
        if dist_raced > 6400:
            print("🏁 Koniec pokazu.")
            done = True

if __name__ == "__main__":
    play()