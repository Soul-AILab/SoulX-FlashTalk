import gradio as gr
import os
import torch
import numpy as np
import time
import imageio
import librosa
import subprocess
from datetime import datetime
from collections import deque
from loguru import logger
import argparse

# Import internal modules
from flash_talk.inference import get_pipeline, get_base_data, get_audio_embedding, run_pipeline, infer_params

# Global variable to store the loaded pipeline
pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None

def run_multi_gpu_inference(
    gpu_ids,
    ckpt_dir,
    wav2vec_dir,
    input_prompt,
    cond_image,
    audio_path,
    audio_encode_mode,
    seed,
    progress=gr.Progress()
):
    """
    Executes the inference using torchrun for Multi-GPU support.
    """
    gpu_list = [x.strip() for x in gpu_ids.split(',') if x.strip()]
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        raise gr.Error("Please specify at least one GPU ID (e.g., '0,1,2,3').")

    cuda_visible_devices = ",".join(gpu_list)
    
    # Define output path beforehand to know where to look
    output_dir = 'gradio_results_multigpu'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    # Note: generate_video.py generates its own filename, so we pass --save_file to control it
    filename = f"res_{timestamp}.mp4"
    save_path = os.path.abspath(os.path.join(output_dir, filename))

    # Construct the command
    # CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... generate_video.py ...
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "generate_video.py",
        "--ckpt_dir", ckpt_dir,
        "--wav2vec_dir", wav2vec_dir,
        "--input_prompt", input_prompt,
        "--cond_image", cond_image,
        "--audio_path", audio_path,
        "--audio_encode_mode", audio_encode_mode,
        "--base_seed", str(int(seed)),
        "--save_file", save_path
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    logger.info(f"Starting Multi-GPU inference with command: {' '.join(cmd)}")
    logger.info(f"Visible Devices: {cuda_visible_devices}")
    
    progress(0, desc="Starting Multi-GPU process... (Check terminal for logs)")
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Read output line by line to update progress (simple heuristic)
        for line in process.stdout:
            print(line, end='') # Print to console for debugging
            if "Generate video chunk" in line:
                progress(0.5, desc="Generating chunks...")
            elif "Saving generated video" in line:
                progress(0.9, desc="Saving video...")
        
        process.wait()
        
        if process.returncode != 0:
            raise gr.Error(f"Multi-GPU process failed with return code {process.returncode}")
            
    except Exception as e:
        logger.error(f"Error during multi-gpu execution: {e}")
        raise gr.Error(f"Multi-GPU execution failed: {e}")

    if os.path.exists(save_path):
        return save_path
    else:
        raise gr.Error("Output video file was not found. Check terminal logs for errors.")

def save_video_to_file(frames_list, video_path, audio_path, fps):
    """
    Helper function to save the video, similar to generate_video.py but adapted for function usage.
    """
    temp_video_path = video_path.replace('.mp4', '_temp.mp4')
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    try:
        with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                                fps=fps , codec='h264', ffmpeg_params=['-bf', '0']) as writer:
            for frames in frames_list:
                frames = frames.numpy().astype(np.uint8)
                for i in range(frames.shape[0]):
                    frame = frames[i, :, :, :]
                    writer.append_data(frame)
        
        # merge video and audio
        # Use aac audio codec for better compatibility instead of copy
        # This handles cases where input audio (like PCM wav) is not supported in MP4 container
        cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    return video_path

def run_inference(
    ckpt_dir,
    wav2vec_dir,
    input_prompt,
    cond_image,
    audio_path,
    audio_encode_mode,
    seed,
    cpu_offload,
    progress=gr.Progress()
):
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir

    # 1. Load Model if needed
    if pipeline is None or loaded_ckpt_dir != ckpt_dir or loaded_wav2vec_dir != wav2vec_dir:
        progress(0, desc="Loading Model...")
        logger.info(f"Loading pipeline with ckpt_dir={ckpt_dir}, wav2vec_dir={wav2vec_dir}")
        try:
            pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, cpu_offload=cpu_offload)
            loaded_ckpt_dir = ckpt_dir
            loaded_wav2vec_dir = wav2vec_dir
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise gr.Error(f"Failed to load model: {e}")

    # 2. Prepare Data
    progress(0.1, desc="Preparing Data...")
    
    # Handle seed
    base_seed = int(seed) if seed >= 0 else 9999
    
    # Get parameters from global config (infer_params)
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    # Prepare base data (prompt, image)
    try:
        get_base_data(pipeline, input_prompt=input_prompt, cond_image=cond_image, base_seed=base_seed)
    except Exception as e:
        logger.error(f"Error in get_base_data: {e}")
        raise gr.Error(f"Error processing inputs: {e}")

    generated_list = []
    
    # Load Audio
    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio file: {e}")

    logger.info("Data preparation done. Start to generate video...")

    # 3. Generation Loop
    if audio_encode_mode == 'once':
        audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
        audio_embedding_chunks_list = [audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous() for i in range((audio_embedding_all.shape[1]-frame_num) // slice_len)]
        
        total_chunks = len(audio_embedding_chunks_list)
        for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
            progress(0.2 + 0.7 * (chunk_idx / total_chunks), desc=f"Generating chunk {chunk_idx+1}/{total_chunks}")
            
            torch.cuda.synchronize()
            start_time = time.time()

            # inference
            video = run_pipeline(pipeline, audio_embedding_chunk)

            torch.cuda.synchronize()
            end_time = time.time()
            logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")
            
            generated_list.append(video.cpu())

    elif audio_encode_mode == 'stream':
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_slices = human_speech_array_all[:(len(human_speech_array_all)//(human_speech_array_slice_len))*human_speech_array_slice_len].reshape(-1, human_speech_array_slice_len)

        total_chunks = len(human_speech_array_slices)
        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            progress(0.2 + 0.7 * (chunk_idx / total_chunks), desc=f"Generating chunk {chunk_idx+1}/{total_chunks}")
            
            # streaming encode audio chunks
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

            torch.cuda.synchronize()
            start_time = time.time()

            # inference
            video = run_pipeline(pipeline, audio_embedding)

            torch.cuda.synchronize()
            end_time = time.time()
            logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")

            generated_list.append(video.cpu())

    # 4. Save Video
    progress(0.95, desc="Saving Video...")
    output_dir = 'gradio_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    filename = f"res_{timestamp}.mp4"
    save_path = os.path.join(output_dir, filename)
    
    final_video_path = save_video_to_file(generated_list, save_path, audio_path, fps=tgt_fps)
    logger.info(f"Saved to {final_video_path}")
    
    return final_video_path

# Gradio Interface Definition
with gr.Blocks(title="SoulX-FlashTalk Video Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ SoulX-FlashTalk Video Generator")
    gr.Markdown("Upload an image and an audio file to generate a talking head video.")

    with gr.Row():
        with gr.Column(scale=1):
            # 1. Main Inputs Section (Always Visible)
            with gr.Group():
                gr.Markdown("### 🎬 Generation Inputs")
                with gr.Row():
                    cond_image_input = gr.Image(
                        label="Condition Image", 
                        type="filepath", 
                        value="examples/man.png",
                        height=300
                    )
                    audio_path_input = gr.Audio(
                        label="Audio Input", 
                        type="filepath", 
                        value="examples/cantonese_16k.wav"
                    )
                
                input_prompt_input = gr.Textbox(
                    label="Input Prompt", 
                    value="A person is talking. Only the foreground characters are moving, the background remains static.",
                    lines=2,
                    placeholder="Describe the motion..."
                )

            # 2. Main Action Button
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

            # 3. Advanced Configuration (Collapsed by default to save space)
            with gr.Accordion("⚙️ Advanced Settings & Model Configuration", open=False):
                with gr.Tabs():
                    with gr.TabItem("Execution Mode"):
                        mode_input = gr.Radio(
                            choices=["Single GPU (Interactive)", "Multi-GPU (Batch Process)"],
                            value="Single GPU (Interactive)",
                            label="Execution Mode",
                            info="Single GPU: Keeps model in memory for fast interactive use. Multi-GPU: Spawns new process, better for stability/isolation."
                        )
                        gpu_ids_input = gr.Textbox(
                            label="GPU IDs (Multi-GPU only)",
                            value="0,1,2,3",
                            visible=False,
                            placeholder="0,1,2,3"
                        )
                        cpu_offload_input = gr.Checkbox(
                            label="Enable CPU Offload", 
                            value=True,
                            info="Enable CPU offload for low VRAM(48G) usage (Single GPU only)."
                        )

                    with gr.TabItem("Model Paths"):
                        ckpt_dir_input = gr.Textbox(
                            label="FlashTalk Checkpoint Directory", 
                            value="models/SoulX-FlashTalk-14B",
                            info="Path to the FlashTalk model checkpoint."
                        )
                        wav2vec_dir_input = gr.Textbox(
                            label="Wav2Vec Directory", 
                            value="models/chinese-wav2vec2-base",
                            info="Path to the Wav2Vec model checkpoint."
                        )

                    with gr.TabItem("Inference Params"):
                        audio_encode_mode_input = gr.Radio(
                            label="Audio Encode Mode", 
                            choices=["stream", "once"], 
                            value="stream",
                            info="Stream: chunk-by-chunk; Once: all at once."
                        )
                        seed_input = gr.Number(
                            label="Random Seed", 
                            value=9999, 
                            precision=0
                        )

        with gr.Column(scale=1):
            gr.Markdown("### 📺 Output Video")
            video_output = gr.Video(label="Generated Video", height=500)

    # Event Handlers
    def update_visibility(mode):
        if mode == "Single GPU (Interactive)":
            return gr.update(visible=False), gr.update(visible=True, value=True)
        else:
            return gr.update(visible=True), gr.update(visible=False, value=False)

    mode_input.change(fn=update_visibility, inputs=mode_input, outputs=[gpu_ids_input, cpu_offload_input])

    def dispatch_inference(
        mode, gpu_ids, ckpt, wav2vec, prompt, img, audio, enc_mode, seed, cpu_off
    ):
        if mode == "Single GPU (Interactive)":
            return run_inference(ckpt, wav2vec, prompt, img, audio, enc_mode, seed, cpu_off)
        else:
            return run_multi_gpu_inference(gpu_ids, ckpt, wav2vec, prompt, img, audio, enc_mode, seed)

    # Event Binding
    generate_btn.click(
        fn=dispatch_inference,
        inputs=[
            mode_input,
            gpu_ids_input,
            ckpt_dir_input,
            wav2vec_dir_input,
            input_prompt_input,
            cond_image_input,
            audio_path_input,
            audio_encode_mode_input,
            seed_input,
            cpu_offload_input
        ],
        outputs=video_output
    ) 

if __name__ == "__main__":
    app.launch()
