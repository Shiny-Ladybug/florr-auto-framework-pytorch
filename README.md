# florr-auto-framework-pytorch

> [!IMPORTANT]  
> If you are not a Pro Developer or a enthusiast, THIS REPO IS NOT WHAT YOU WANT

This is a DEMO repo to show how to train a **custom florr-ai model**.

The demo code is for **Starfish Zone**.

The code is highly hard to run by yourself.

However, it will sure amaze you if you deploy and run it successfully, and I'll invite you to be the collaborator if you succeed 

Leave your message in Issue page to let me know.

## Training

In dataset_utils.py: 208

```python
class FlorrModel(nn.Module):
    def __init__(self, input_dim=73, output_dim=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x[:, 0:2] = self.tanh(x[:, 0:2])  # Move output
        x[:, 2:5] = self.sigmoid(x[:, 2:5])  # Attack, Defend, YinYang output
        return x
```

You can add / remove the layer of NN when necessary.

For example, here's a version of `Attention-Layer`

```python

class FlorrModel(nn.Module):
    def __init__(self, input_dim=73, output_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc3 = nn.Linear(64, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # Add attention: treat each sample as a sequence of length 1
        # Expand x to (batch, seq_len=1, features=64)
        x_seq = x.unsqueeze(1)
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        x = attn_output.squeeze(1)
        x = self.fc3(x)
        x[:, 0:2] = self.tanh(x[:, 0:2])  # Move output
        x[:, 2:5] = self.sigmoid(x[:, 2:5])  # Attack, Defend, YinYang output
        return x
```
