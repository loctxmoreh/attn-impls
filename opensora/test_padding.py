import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.nn import functional as F

from pt_impl import pt_padded

q = torch.randn(60, 16, 3600, 72)
k = torch.randn(60, 16, 3600, 72)
v = torch.randn(60, 16, 3600, 72)

out1 = F.scaled_dot_product_attention(q, k, v)
out2 = pt_padded(q, k, v)

print(out1[0, 0, 0])
print(out2[0, 0, 0])

print(torch.max(torch.abs(out1 - out2)))
print(torch.allclose(out1, out2, rtol=0.01, atol=0.001))
