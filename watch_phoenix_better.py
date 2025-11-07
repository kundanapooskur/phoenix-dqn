import gymnasium as gym
import ale_py
import random
import time

gym.register_envs(ale_py)

env = gym.make("ALE/Phoenix-v5", render_mode="human")

print("�� Phoenix Demo - Mixed Random/Pattern Actions")
print("This will show more varied gameplay")
print("Close window to stop")

obs, _ = env.reset()
total_reward = 0
frame_count = 0

while True:
    # Create more interesting movement patterns
    frame_count += 1
    
    # Change strategy every 30 frames
    pattern = (frame_count // 30) % 4
    
    if pattern == 0:
        # Move and shoot right
        action = random.choice([3, 6])  # RIGHT or RIGHTFIRE
    elif pattern == 1:
        # Move and shoot left  
        action = random.choice([4, 7])  # LEFT or LEFTFIRE
    elif pattern == 2:
        # Just fire
        action = 1  # FIRE
    else:
        # Random action
        action = env.action_space.sample()
    
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    if reward > 0:
        print(f"Hit! +{reward} points")
    
    time.sleep(0.02)
    
    if done or truncated:
        print(f"Game Over! Score: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0
        frame_count = 0

env.close()
