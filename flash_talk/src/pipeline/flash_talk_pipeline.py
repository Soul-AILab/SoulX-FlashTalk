# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import types
from PIL import Image
from loguru import logger
import time

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor

from flash_talk.wan.modules import (CLIPModel, T5EncoderModel, WanVAE)
from flash_talk.infinite_talk.modules.multitalk_model import WanModel
from flash_talk.infinite_talk.audio_analysis.wav2vec2 import Wav2Vec2Model
from flash_talk.infinite_talk.utils.multitalk_utils import match_and_blend_colors_torch, resize_and_centercrop

# compile models to speedup inference
COMPILE_MODEL = True
COMPILE_VAE = True
# use parallel vae to speedup decode/encode
USE_PARALLEL_VAE = True

def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                module._buffers[name] = buf.to(param_dtype)

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class FlashTalkPipeline:
    def __init__(
        self,
        config,
        checkpoint_dir,
        wav2vec_dir,
        device="cuda",
        use_usp=False,
        num_timesteps=1000,
        use_timestep_transform=True,
    ):
        r"""
        Initializes the image-to-video generation model components.
        Reference from InfiniteTalkPipeline: https://github.com/MeiGen-AI/InfiniteTalk/blob/main/wan/multitalk.py
        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            wav2vec_dir (`str`):
                Path to directory containing wav2vec checkpoints
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
        """
        self.device = device
        self.config = config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.use_usp = use_usp and dist.is_initialized()
        self.param_dtype = config.param_dtype

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        
        self.vae = WanVAE(
            vae_path=os.path.join(checkpoint_dir, config.vae_checkpoint),
            # dtype=dtype,
            device=self.device,
            parallel=(USE_PARALLEL_VAE and self.use_usp),
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logger.info(f"Creating WanModel from {checkpoint_dir}")

        self.model = WanModel.from_pretrained(
            checkpoint_dir,
            device_map=self.device,
            torch_dtype=self.param_dtype,
        )

        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from ...infinite_talk.distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        if COMPILE_MODEL:
            self.model = torch.compile(self.model)
        if COMPILE_VAE:
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)

        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)

    @torch.no_grad()
    def prepare_params(self,
                        input_prompt,
                        cond_image,
                        target_size,
                        frame_num,
                        motion_frames_num,
                        sampling_steps,
                        seed=None,
                        shift=5.0,
                        color_correction_strength=0.0,
                        ):

        context = self.text_encoder([input_prompt], self.device)[0]

        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num

        self.target_h, self.target_w = target_size
        self.lat_h, self.lat_w = self.target_h // self.vae_stride[1], self.target_w // self.vae_stride[2]

        if isinstance(cond_image, str):
            cond_image = Image.open(cond_image).convert("RGB")
        cond_image_tensor = resize_and_centercrop(cond_image, (self.target_h, self.target_w)).to(self.device)
        cond_image_tensor = (cond_image_tensor / 255 - 0.5) * 2

        self.cond_image_tensor = cond_image_tensor

        self.color_correction_strength = color_correction_strength
        self.original_color_reference = None
        if self.color_correction_strength > 0.0:
            self.original_color_reference = cond_image_tensor.clone()

        clip_context = self.clip.visual(cond_image_tensor[:, :, -1:, :, :]).to(self.param_dtype)
        video_frames = torch.zeros(1, cond_image_tensor.shape[1], frame_num-cond_image_tensor.shape[2], self.target_h, self.target_w).to(self.device)

        padding_frames_pixels_values = torch.concat([cond_image_tensor, video_frames], dim=2)
        y = self.vae.encode(padding_frames_pixels_values)
        common_y = y.unsqueeze(0).to(self.param_dtype)

        # get mask
        msk = torch.ones(1, frame_num, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2).to(self.param_dtype)

        y = torch.concat([msk, common_y], dim=1)


        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # prepare timesteps
        if sampling_steps == 2:
            timesteps = [1000, 500]
        elif sampling_steps == 4:
            timesteps = [1000, 750, 500, 250]
        else:
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            
        timesteps.append(0.)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
        self.timesteps = timesteps

        self.arg_c = {
            'context': [context],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': y,
            'ref_target_masks': None,
        }

        self.latent_motion_frames = self.vae.encode(self.cond_image_tensor)

        return

    @torch.no_grad()
    def preprocess_audio(self, speech_array, sr=16000, fps=25):
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * fps

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            logger.error("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb

    @torch.no_grad()
    def generate(self, audio_embedding):

        # evaluation mode
        with torch.no_grad():

            self.arg_c.update({
                "audio": audio_embedding,
            })

            # sample videos
            latent = torch.randn(
                16, (self.frame_num - 1) // 4 + 1,
                self.lat_h,
                self.lat_w,
                dtype=torch.float32,
                device=self.device,
                generator=self.generator)

            latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            for i in range(len(self.timesteps)-1):
                timestep = self.timesteps[i]
                latent_model_input = [latent]

                torch.cuda.synchronize()
                start_time = time.time()

                # inference without CFG
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **self.arg_c)[0]

                torch.cuda.synchronize()
                end_time = time.time()
                if self.rank == 0:
                    print(f'[generate] model denoise per step: {end_time - start_time}s')

                noise_pred = -noise_pred_cond

                # update latent
                t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
                t_i_1 = self.timesteps[i+1][:, None, None, None] / self.num_timesteps
                x_0 = latent + noise_pred * t_i

                latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator)

                latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            torch.cuda.synchronize()
            start_decode_time = time.time()
            videos = self.vae.decode(latent)
            torch.cuda.synchronize()
            end_decode_time = time.time()
            if self.rank == 0:
                print(f'[generate] decode video frames: {end_decode_time - start_decode_time}s')
        
        torch.cuda.synchronize()
        start_color_correction_time = time.time()
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(videos, self.original_color_reference, self.color_correction_strength)

        cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device).to(torch.float32)
        torch.cuda.synchronize()
        end_color_correction_time = time.time()
        if self.rank == 0:
            print(f'[generate] color correction: {end_color_correction_time - start_color_correction_time}s')

        torch.cuda.synchronize()
        start_encode_time = time.time()
        self.latent_motion_frames = self.vae.encode(cond_frame)
        torch.cuda.synchronize()
        end_encode_time = time.time()
        if self.rank == 0:
            print(f'[generate] encode motion frames: {end_encode_time - start_encode_time}s')

        gen_video_samples = videos[:, :, self.motion_frames_num:]

        return gen_video_samples[0]
