
import torch
import time

N = 100000

a = torch.sin(torch.arange(N, device='cuda', dtype=torch.float64))
b = torch.cos(torch.arange(N, device='cuda', dtype=torch.float64))

c = torch.empty_like(a)

c[:] = a + b

torch.cuda.synchronize()
start = time.time()

c[:] = a + b

torch.cuda.synchronize()
end = time.time()

result = c.cpu().numpy()
print(f"sum/n = {result.sum() / N} (should be ~1)")

print(f"Time: {(end - start)*1e6:.2f} µs")