import torch
import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)

class PhoenixDQN(torch.nn.Module):
    def __init__(self, n_actions=8):
        super(PhoenixDQN, self).__init__()
        self.fc1 = torch.nn.Linear(210*160*3, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = x.flatten(start_dim=1) / 255.0
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

model = PhoenixDQN()
model.load_state_dict(torch.load('phoenix_model.pth'))
model.eval()

env = gym.make("ALE/Phoenix-v5", render_mode="human")

print("🎮 Watch your trained Phoenix agent play!")
print("Close window to stop...")

obs, _ = env.reset()
total_reward = 0

while True:
    with torch.no_grad():
        state = torch.FloatTensor(obs).unsqueeze(0)
        action = model(state).argmax().item()
    
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    time.sleep(0.01)
    
    if done or truncated:
        print(f"Game Over! Score: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0

env.close()
