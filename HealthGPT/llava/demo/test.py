import torch
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# 检查 GPU 是否可用
if not torch.cuda.is_available():
    raise RuntimeError("GPU is not available. FlashAttention requires a GPU.")

# 创建一个简单的张量
qkv = torch.randn(1, 256, 3, 8, 64, device="cuda")  # (batch_size, seqlen, 3, nheads, headdim)

# 检查数据类型
print("Original dtype:", qkv.dtype)

# 转换为 fp16
qkv = qkv.to(torch.float16)

# 再次检查数据类型
print("Converted dtype:", qkv.dtype)

# 使用 flash_attn_qkvpacked_func
out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)
print("FlashAttention output shape:", out.shape)