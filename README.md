<div align="center">


<h1>SoulX-FlashTalk: Real-Time Infinite Streaming of Audio-Driven Avatars via Self-Correcting Bidirectional Distillation</h1>

[Le Shen*](https://openreview.net/profile?id=%7ELe_Shen3), [Qian Qiao*](https://qianqiaoai.github.io/), [Tan Yu*](https://jiayoujiayoujiayoua.github.io/), [Ke Zhou](https://github.com/jokerz0624), [Tianhang Yu](#), [Yu Zhan](#),  [Zhenjie Wang](#), [Dingcheng Zhen](#), [Ming Tao](#), [Shunshun Yin](#), [Siyuan Liu](#) <sup>&#9993;</sup>



<sup>*</sup>Equal Contribution
<sup>&#9993;</sup>Corresponding Author


<a href='https://soul-ailab.github.io/soulx-flashtalk/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2512.23379'><img src='https://img.shields.io/badge/Technical-Report-red'></a>
<a href="https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Spaces-blue" alt="HF space"></a>&nbsp;
<a href='https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>


## 🔥 News
- **2026.02.12** - We have released the [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead), which is a streaming talking head project that achieves real-time performance on consumer GPUs (e.g., RTX 4090/5090).
- **2026.01.08** - We have released the [inference code](https://github.com/Soul-AILab/SoulX-FlashTalk), and the [model weights](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B).
- **2025.12.30** - We released **Project page** on [SoulX-FlashTalk](https://soul-ailab.github.io/soulx-flashtalk/).
- **2025.12.30** - We released **SoulX-FlashTalk Technical Report** on [Arxiv](https://arxiv.org/pdf/2512.23379) and [GitHub repository](./assets/SoulX_FlashTalk.pdf).

## 🤫 Coming soon
**A 4-GPU real-time version of SoulX-FlashTalk.**

## 📑 Todo List
- [x] Technical report 
- [x] Project Page
- [x] Inference code
- [x] Checkpoint release
- [ ] Online demo

## 📢 Live Streaming & Video Podcast

<p align="center">
  <video src="https://private-user-images.githubusercontent.com/176391424/542734488-c2c68ca1-5ac1-431f-9783-7af6e20e243e.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk3NjU2NzUsIm5iZiI6MTc2OTc2NTM3NSwicGF0aCI6Ii8xNzYzOTE0MjQvNTQyNzM0NDg4LWMyYzY4Y2ExLTVhYzEtNDMxZi05NzgzLTdhZjZlMjBlMjQzZS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEzMFQwOTI5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hNGZmZTY1OGRlZjRiNTBiM2Y2YjlhM2E3MWZhYmVhZDIxMDI3ZGFmZmY1NmVjZDgzNzVkODQ1YjM1Y2M0NmEzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.boX3RzfPeOTMhKwt9ZLzsD8sgEJep_OEGDvP7fnqzTA" style="width:100%; max-width:640px; aspect-ratio:16/9; object-fit:cover;" controls loop></video>
</p>

## 🎬 Online Demos
<table>
  <tbody>
    <tr>
      <td width="50%"><video src="https://private-user-images.githubusercontent.com/176391424/542681016-8405bc16-836d-4497-aa86-b62e9fc7dbed.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk3NjAyNDUsIm5iZiI6MTc2OTc1OTk0NSwicGF0aCI6Ii8xNzYzOTE0MjQvNTQyNjgxMDE2LTg0MDViYzE2LTgzNmQtNDQ5Ny1hYTg2LWI2MmU5ZmM3ZGJlZC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEzMFQwNzU5MDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kODAzMzcyZjQzZDg3N2U1YjJlNDZhOTE4ZTllMGRlOTY4OWZhOWVjOWZjNjc1N2QwNWU5OGQ2ZWVjMzY0YWYxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.3oKK0XwqanMJZy6iARStVaPbMRcpQWeTrRlCNQeSXl0" style="width:100%; aspect-ratio:16/9; object-fit:cover;" controls loop></video></td>
      <td width="50%"><video src="https://private-user-images.githubusercontent.com/176391424/542682295-9e59ad13-7d3a-4bfc-a949-1a00da5d19f4.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk3NjAyNTksIm5iZiI6MTc2OTc1OTk1OSwicGF0aCI6Ii8xNzYzOTE0MjQvNTQyNjgyMjk1LTllNTlhZDEzLTdkM2EtNGJmYy1hOTQ5LTFhMDBkYTVkMTlmNC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEzMFQwNzU5MTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jOGNhMDcwZGUyYjdkMzM5YTZmYjFjMTMxZDQ0MTI5YmVmNjEzMWNmMjJmZTkyMjc5NWRmMTQzMGIzNjhjYjAyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.9fzD-WHHZPl5lm6W4S-cYU5giKsid0wCQj4qpkGI838" style="width:100%; aspect-ratio:16/9; object-fit:cover;" controls loop></video></td>
    </tr>
  </tbody>
</table>

## 🌰 Examples


<table>
  <tbody>
    <!-- Row 1: Videos 1-5 -->
    <tr>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/536123542-cee5c716-3267-42d9-86c0-93de8e9ed7fa.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg0Njk5NDIsIm5iZiI6MTc2ODQ2OTY0MiwicGF0aCI6Ii8xNzYzOTE0MjQvNTM2MTIzNTQyLWNlZTVjNzE2LTMyNjctNDJkOS04NmMwLTkzZGU4ZTllZDdmYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExNVQwOTM0MDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04NjhjZjYxYmJkZTE2M2I4YTVhNWNhN2U5ZDBhZGM0Yzc2NGM4YjA0ZDQ4NGIzMGEzZjVmZGIzOWZmYTI4Mzg1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.8fNUNx1Io8JvBghhQAC7mmHHa_oF3ajAd-cOeXnoGwI" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src=https://private-user-images.githubusercontent.com/176391424/536124198-2ce79455-edb7-4fba-8522-dc9448ddb37a.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg0Njk5MzksIm5iZiI6MTc2ODQ2OTYzOSwicGF0aCI6Ii8xNzYzOTE0MjQvNTM2MTI0MTk4LTJjZTc5NDU1LWVkYjctNGZiYS04NTIyLWRjOTQ0OGRkYjM3YS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExNVQwOTMzNTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lNTgyMWMxNWU2MjI0YjQwNmJmM2Y4MWVlYjc3OTlmMzZjNTg2OGI5MTlhODExMDQ5M2E5NGNhOGJjMTgzMDllJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.0peFNyujNHJ0n8xMyj19VjTalPV74lE6oepewP-pX0A" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/536126414-de649e5f-b09a-408d-9bff-96574326285c.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg0Njk5MzksIm5iZiI6MTc2ODQ2OTYzOSwicGF0aCI6Ii8xNzYzOTE0MjQvNTM2MTI2NDE0LWRlNjQ5ZTVmLWIwOWEtNDA4ZC05YmZmLTk2NTc0MzI2Mjg1Yy5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExNVQwOTMzNTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNTQ1Mjg0ZWUwZmIyYzQ2OTkxYzY5ZjZmYjRjYmU0MDA0Yzg3YTgwZDEwYWM4YTIzNmFlMDhkZGVlNDI0N2U3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.6VE5cy4RuvPe49zuT6OHjo6XILn17QYT9kfS7_X3Efw" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
    </tr>

  </tbody>
</table>



## 📖 Quickstart
###  🔧 Installation
#### 1. Create a Conda environment
```bash
conda create -n flashtalk python=3.10
conda activate flashtalk
```
#### 2. Install PyTorch on CUDA
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
#### 3. Install other dependencies
```bash
pip install -r requirements.txt
```
#### 4. Flash-attention installation:
```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```
#### 5. FFmpeg installation
```bash
# Ubuntu / Debian
apt-get install ffmpeg
# CentOS / RHEL
yum install ffmpeg ffmpeg-devel
```
or
```bash
# Conda (no root required) 
conda install -c conda-forge ffmpeg==7
```
### 🤗 Model download
| Model Component | Description | Link |
| :--- | :--- | :---: |
| `SoulX-FlashTalk-14B` | Our 14b model| 🤗 [Huggingface](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B) |
| `chinese-wav2vec2-base` | chinese-wav2vec2-base | 🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |

```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashTalk-14B --local-dir ./models/SoulX-FlashTalk-14B
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./models/chinese-wav2vec2-base
```
### 🚀 Inference
```bash
# Infer on single GPU
# Requires more than 64G of VRAM. Use --cpu_offload to reduce VRAM usage to 40G.
bash inference_script_single_gpu.sh

# Infer on multy GPUs
# Real-time inference speed can only be supported on 8xH800 or higher graphics cards
bash inference_script_multi_gpu.sh
```

### 👋 Online Demo 
Coming Soon!


## 📧 Contact Us
<!-- If you are interested in leaving a message to our work, feel free to email le.shen@mail.dhu.edu.cn or qiaoqian@soulapp.cn or yutan@soulapp.cn or zhouke@soulapp.cn or liusiyuan@soulapp.cn

You’re welcome to join our WeChat group for technical discussions, updates.

Due to Group 1 reaching its capacity, we have opened a new WeChat group for further technical discussions and updates. Feel free to join us!
<p align="center">
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/wechat_group.png" width="300" alt="WeChat Group QR Code"/>
  </span>
</p> -->
If you are interested in leaving a message to our work, feel free to email le.shen@mail.dhu.edu.cn or qiaoqian@soulapp.cn or yutan@soulapp.cn or zhouke@soulapp.cn or liusiyuan@soulapp.cn

Due to Group 1 reaching its capacity, we have opened a new WeChat group. Additionally, we represent **SoulApp** and warmly welcome everyone to download the app and join our Soul group for further technical discussions and updates!

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/wechat_group.png" width="300" alt="WeChat Group QR Code"/>
        <br />
        <strong>Join WeChat Group<br>(加入微信技术群)</strong>
      </td>
      <td width="100"></td>
      <td align="center">
        <img src="assets/soul_group.png" width="300" alt="Soul App Group QR Code"/>
        <br />
        <strong>Download SoulApp & Join Group<br>(下载SoulApp加入群组)</strong>
      </td>
    </tr>
  </table>
</div>

 
## 📚 Citation

If you find our work useful in your research, please consider citing:

```
@misc{shen2025soulxflashtalk,
  title = {{SoulX-FlashTalk}: Real-Time Infinite Streaming of Audio-Driven Avatars via Self-Correcting Bidirectional Distillation},
  author = {Shen, Le and Qiao, Qian and Yu, Tan and Zhou, Ke and Yu, Tianhang and Zhan, Yu and Wang, Zhenjie and Tao, Ming and Yin, Shunshun and Liu, Siyuan},
  year = {2025},
  eprint = {2512.23379},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2512.23379},
  url = {https://arxiv.org/abs/2512.23379}
}
```

## 🙇 Acknowledgement
- [Infinitetalk](https://github.com/MeiGen-AI/InfiniteTalk) and [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon.
- [Self forcing](https://github.com/guandeh17/Self-Forcing): the codebase we built upon.
- [DMD](https://github.com/tianweiy/DMD2) and [Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus): the key distillation technique used by our method.
> [!TIP]
> If you find our work useful, please also consider starring the original repositories of these foundational methods.

## 💡 Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=Soul-AILab/SoulX-FlashTalk&type=date&legend=top-left)](https://star-history.dera.page/#Soul-AILab/SoulX-FlashTalk&type=date&legend=top-left)
