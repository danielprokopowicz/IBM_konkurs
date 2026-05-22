# Team Sqro — IBM AI Racing League

AI racing driver for TORCS Corkscrew track using Behavioral Cloning.

## How it works
- `collect` — rule-based driver collects 30 laps of training data
- `bc` — neural network trained via Behavioral Cloning (steer + accel + brake)  
- `play` — AI drives autonomously, code handles gears only

## Architecture
- Input: 14 features (angle, track position, speed, 9 track sensors)
- Hidden layers: 256 → 128 → 64 neurons (ReLU)
- Output: steer (tanh), accel (sigmoid), brake (sigmoid)
- Training: 500 epochs, Adam optimizer, MSE loss = 0.0013

## Usage
python train.py collect   # Phase 1: collect training data
python train.py bc        # Phase 2: Behavioral Cloning
python train.py play      # Run trained AI driver

## Results
Lap time: ~1:42 from standing start on Corkscrew track

## Team
Team Sqro — Silesian University of Technology  
IBM AI Racing League 2026
