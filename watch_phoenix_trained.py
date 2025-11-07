import torch
import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)

class PhoenixDQN(torch.nn.Module):
    def __init__(self, n_actions=8):  # Phoenix has 8 actions
        super(PhoenixDQN, self).__init__()
        self.fc1 = torch.nn.Linear(210*160*3, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = x.flatten(start_dim=1) / 255.0
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Load your trained Phoenix model
try:
    # Try different possible model names
    try:
        model = PhoenixDQN()
        model.load_state_dict(torch.load('phoenix_model.pth'))
        print("Loaded phoenix_model.pth")
    except:
        try:
            model = PhoenixDQN()
            model.load_state_dict(torch.load('phoenix_fast1000.pth'))
            print("Loaded phoenix_fast1000.pth")
        except:
            model = PhoenixDQN()
            model.load_state_dict(torch.load('phoenix_checkpoint_100.pth'))
            print("Loaded checkpoint")
    
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load trained model: {e}")
    print("Using untrained model for demonstration")
    model = PhoenixDQN()
    model.eval()

# Create Phoenix with visual rendering
env = gym.make("ALE/Phoenix-v5", render_mode="human")

print("\n🎮 Watching Phoenix Agent Play!")
print("Close the game window to stop...")
print("-" * 40)

obs, _ = env.reset()
total_reward = 0
episode_count = 0

while True:
    # Use model to select action
    with torch.no_grad():
        state = torch.FloatTensor(obs).unsqueeze(0)
        q_values = model(state)
        action = q_values.argmax().item()
    
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    time.sleep(0.01)  # Slight delay to make it watchable
    
    if done or truncated:
        episode_count += 1
        print(f"Episode {episode_count} | Score: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0

env.close()
