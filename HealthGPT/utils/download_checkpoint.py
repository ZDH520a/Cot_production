from huggingface_hub import snapshot_download

# 下载 HealthGPT-M3
healthgpt_m3_path = snapshot_download(repo_id="lintw/HealthGPT-M3", cache_dir="./models/HealthGPT-M3")
print(f"HealthGPT-M3 downloaded to: {healthgpt_m3_path}")

# 下载 CLIP ViT Large Patch14-336
clip_path = snapshot_download(repo_id="openai/clip-vit-large-patch14-336", cache_dir="./models/clip-vit-large-patch14-336")
print(f"CLIP-ViT Large Patch14-336 downloaded to: {clip_path}")

# 下载 Phi-3-mini-4k-instruct
phi_path = snapshot_download(repo_id="microsoft/Phi-3-mini-4k-instruct", cache_dir="./models/Phi-3-mini-4k-instruct")
print(f"Phi-3-mini4k-instruct downloaded to: {phi_path}")
