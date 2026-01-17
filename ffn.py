import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GEGLU_FFN(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=12288):
        super().__init__()
        self.Wu = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wv = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        u = self.Wu(x)      # (B, 12288)
        v = self.Wv(x)      # (B, 12288)
        g = F.gelu(u)
        h = g * v
        return self.Wo(h)   # (B, 4096)

device = "cuda"
print("Device:", device)

ffn = GEGLU_FFN().to(device)
batch_sizes = [4, 8, 16, 32, 64, 128]

# Warm-up
for B in batch_sizes:
    x = torch.randn(B, 4096, device=device)
    for _ in range(5):
        _ = ffn(x)

for B in batch_sizes:
    x = torch.randn(B, 4096, device=device)

    # GPU sync before timing
    torch.cuda.synchronize()
    start = time.perf_counter()

    y = ffn(x)
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"Batch {B:>3}: {((end - start) * 1000):.3f} ms")
