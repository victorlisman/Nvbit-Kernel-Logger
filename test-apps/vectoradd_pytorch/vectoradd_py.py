
import torch
import time
import math
import argparse

def main():
    # Parse command line arguments, similar to C version
    parser = argparse.ArgumentParser(description='PyTorch Vector Addition')
    parser.add_argument('n', type=int, nargs='?', default=1024, 
                        help='Size of vectors (default: 1024)')
    args = parser.parse_args()
    n = args.n

    # Allocate and initialize host vectors
    h_a = torch.zeros(n, dtype=torch.float32)
    h_b = torch.zeros(n, dtype=torch.float32)
    h_c = torch.zeros(n, dtype=torch.float32)

    # Initialize vectors on host, matching the C initialization
    for i in range(n):
        h_a[i] = math.sin(i) * math.sin(i)
        h_b[i] = math.cos(i) * math.cos(i)

    # Allocate device memory and copy host vectors to device
    d_a = h_a.cuda()
    d_b = h_b.cuda()
    d_c = torch.zeros(n, dtype=torch.float32, device='cuda')

    # To match the C version's thread block/grid size approach,
    # we'll just use PyTorch's default CUDA kernel execution model
    # PyTorch automatically handles the thread/block configuration

    # Ensure all previous operations are completed
    torch.cuda.synchronize()
    
    # Execute the kernel (vector addition)
    d_c = d_a + d_b
    torch.cuda.synchronize()

    # Copy result back to host
    h_c = d_c.cpu()

    # Sum up vector c and print result divided by n
    sum_c = h_c.sum().item()
    print(f"Final sum = {sum_c}; sum/n = {sum_c / n} (should be ~1)")

    # No need to explicitly free memory in PyTorch as it uses
    # automatic garbage collection, but we can explicitly clear
    # the GPU memory if desired
    del d_a, d_b, d_c
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()