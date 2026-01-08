# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict
#------------------------ Wan MultiTalk AI2V 14B ------------------------#
multitalk_14B = EasyDict(__name__='Config: Wan MultiTalk AI2V 14B')

# t5
multitalk_14B.t5_model = 'umt5_xxl'
multitalk_14B.t5_dtype = torch.bfloat16
multitalk_14B.text_len = 512

# transformer
multitalk_14B.param_dtype = torch.bfloat16

# inference
multitalk_14B.num_train_timesteps = 1000
multitalk_14B.sample_fps = 16
multitalk_14B.sample_neg_prompt = 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'

multitalk_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
multitalk_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
multitalk_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
multitalk_14B.clip_dtype = torch.float16
multitalk_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
multitalk_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
multitalk_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
multitalk_14B.vae_stride = (4, 8, 8)

# transformer
multitalk_14B.patch_size = (1, 2, 2)
multitalk_14B.dim = 5120
multitalk_14B.ffn_dim = 13824
multitalk_14B.freq_dim = 256
multitalk_14B.num_heads = 40
multitalk_14B.num_layers = 40
multitalk_14B.window_size = (-1, -1)
multitalk_14B.qk_norm = True
multitalk_14B.cross_attn_norm = True
multitalk_14B.eps = 1e-6
