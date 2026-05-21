# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import numpy as np
import time
import torch
import torch.distributed as dist
import subprocess
import imageio
import librosa
import numpy as np
from loguru import logger
from collections import deque
from datetime import datetime

from flash_talk.inference import get_pipeline, get_base_data, get_audio_embedding, run_pipeline, infer_params, extract_video_frame, update_cond_image

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify FlashTalk model checkpoint directory."
    assert args.wav2vec_dir is not None, "Please specify the wav2vec checkpoint directory."
    if args.cond_video is not None:
        assert os.path.exists(args.cond_video), f"Condition video not found: {args.cond_video}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else 9999

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to FlashTalk model checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=9999,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--input_prompt",
        type=str,
        default="A person is talking. Only the foreground characters are moving, the background remains static.",
        help="The prompt to generate the video.")
    parser.add_argument(
        "--cond_image",
        type=str,
        default="examples/man.png",
        help="[meta file] The condition image path to generate the video.")
    parser.add_argument(
        "--cond_video",
        type=str,
        default=None,
        help="[optional] A source video path. When provided, each generation chunk "
             "re-anchors appearance to the corresponding frame from this video. "
             "Overrides --cond_image (frame 0 is used as initial conditioning).")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="examples/cantonese_16k.wav",
        help="[meta file] The audio path to generate the video.")
    parser.add_argument(
        "--audio_encode_mode",
        type=str,
        default="stream",
        choices=['stream', 'once'],
        help="stream: encode audio chunk before every generation; once: encode audio together")
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload for low VRAM usage")
    args = parser.parse_args()

    _validate_args(args)

    return args

def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace('res_', '')
    with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                            fps=fps, codec='libx264',
                            ffmpeg_params=['-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p']) as writer:
        for frames in frames_list:
            frames = np.round(frames.numpy()).clip(0, 255).astype(np.uint8)
            for i in range(frames.shape[0]):
                frame = frames[i, :, :, :]
                writer.append_data(frame)
    
    # merge video and audio
    # Use aac audio codec for better compatibility instead of copy
    cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
    subprocess.run(cmd, check=True)
    os.remove(temp_video_path)


def generate(args):
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    pipeline = get_pipeline(world_size=world_size, ckpt_dir=args.ckpt_dir, wav2vec_dir=args.wav2vec_dir, cpu_offload=args.cpu_offload)

    # When a source video is provided, extract frame 0 as initial conditioning image
    initial_cond_image = args.cond_image
    if args.cond_video is not None:
        initial_cond_image = extract_video_frame(args.cond_video, 0)
        if rank == 0:
            logger.info(f"Using video input: {args.cond_video} (frame 0 as initial conditioning)")

    get_base_data(pipeline, input_prompt=args.input_prompt, cond_image=initial_cond_image, base_seed=args.base_seed)

    generated_list = []
    human_speech_array_all, _ = librosa.load(args.audio_path, sr=infer_params['sample_rate'], mono=True)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps


    if rank == 0:
        logger.info("Data preparation done. Start to generate video...")

    if args.audio_encode_mode == 'once':
        # pad audio with silence to avoid truncating the last chunk
        remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate([human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)])

        # encode audio together
        audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)

        # split audio embedding into chunks: 33, 28, 28, 28, ...
        audio_embedding_chunks_list = [audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous() for i in range((audio_embedding_all.shape[1]-frame_num) // slice_len)]

        for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
            torch.cuda.synchronize()
            start_time = time.time()

            # inference
            video = run_pipeline(pipeline, audio_embedding_chunk)

            if chunk_idx != 0:
                video = video[motion_frames_num:]

            torch.cuda.synchronize()
            end_time = time.time()
            if rank == 0:
                logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")

            generated_list.append(video.cpu())

            # Update conditioning image from source video for next chunk
            if args.cond_video is not None:
                next_frame_idx = (chunk_idx + 1) * slice_len
                update_cond_image(pipeline, args.cond_video, next_frame_idx)

    elif args.audio_encode_mode == 'stream':
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        # pad audio with silence to avoid truncating the last chunk
        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate([human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)])

        # split audio embedding into chunks: 28, 28, 28, 28, ...
        human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            # streaming encode audio chunks
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

            torch.cuda.synchronize()
            start_time = time.time()

            # inference
            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:]

            torch.cuda.synchronize()
            end_time = time.time()
            if rank == 0:
                logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")

            generated_list.append(video.cpu())

            # Update conditioning image from source video for next chunk
            if args.cond_video is not None:
                next_frame_idx = (chunk_idx + 1) * slice_len
                update_cond_image(pipeline, args.cond_video, next_frame_idx)


    if rank == 0:
        if args.save_file is None:
            output_dir = 'sample_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")[:-3]
            filename = f"res_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            args.save_file = filepath

        save_video(generated_list, args.save_file, args.audio_path, fps=tgt_fps)
        logger.info(f"Saving generated video to {args.save_file}.mp4")  
        logger.info("Finished.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)