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
- **2025.12.30** - We released **SoulX-FlashTalk Technical Report** on [Arxiv](https://arxiv.org/pdf/2512.23379) and [GitHub repository](./assets/SoulX_FlashTalk.pdf).
- **2025.12.30** - We released **Project page** on [SoulX-FlashTalk](https://soul-ailab.github.io/soulx-flashtalk/).
- **2026.01.08** - We have released the [inference code](https://github.com/Soul-AILab/SoulX-FlashTalk), and the [model weights](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B).

## 🤫 Coming soon
**A 4-GPU version of SoulX-FlashTalk and a new open-source real-time streaming digital human model designed specifically for consumer-grade GPUs.**

## 📑 Todo List
- [x] Technical report 
- [x] Project Page
- [x] Inference code
- [x] Checkpoint release
- [ ] Online demo


## 🎬 Online Demos
<table>
  <tbody>
    <tr>
      <td width="50%"><video src="https://private-user-images.githubusercontent.com/176391424/532641864-17cd4ae6-db1c-4f51-a8bb-2b6cccb0e202.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc3NTQ0NDQsIm5iZiI6MTc2Nzc1NDE0NCwicGF0aCI6Ii8xNzYzOTE0MjQvNTMyNjQxODY0LTE3Y2Q0YWU2LWRiMWMtNGY1MS1hOGJiLTJiNmNjY2IwZTIwMi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwN1QwMjQ5MDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMzQ0NDMxNjQ3ZmFhNjAzMDZkYzBlOWEzOWVkNTM0MDJkYzRmZDBlNDE4YWZmYTY0NzkwY2RhOTVlN2I2ZDM3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.GKmqwpvv46AQIRE89_KouqpC6ZTp3jzqXRIiLTAzxwU" style="width:100%; aspect-ratio:16/9; object-fit:cover;" controls loop></video></td>
      <td width="50%"><video src="https://private-user-images.githubusercontent.com/176391424/532641924-5e2b9ccb-cc62-4250-ae2e-9b7612bf0a04.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc3NTQ0NjAsIm5iZiI6MTc2Nzc1NDE2MCwicGF0aCI6Ii8xNzYzOTE0MjQvNTMyNjQxOTI0LTVlMmI5Y2NiLWNjNjItNDI1MC1hZTJlLTliNzYxMmJmMGEwNC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwN1QwMjQ5MjBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lYWM3YTIzZTM0MTQ3YjdiNTAxMmZkZDNhMjhhNmQ1YjVkNTA4MjQyNzZjY2IxOTY1NmJjZWY4YzgyMjcyZjAyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.AD2JmD_E8syZlQPr5tzGsSoe-o_78SFq77MJ5I7qRAA" style="width:100%; aspect-ratio:16/9; object-fit:cover;" controls loop></video></td>
    </tr>
  </tbody>
</table>

## 🌰 Examples

<!-- ### 1 -->
<table>
  <tbody>
    <!-- Row 1: Videos 1-5 -->
    <tr>
      <td width="20%"><video src="https://private-user-images.githubusercontent.com/176391424/533238933-e6b0e371-4232-4504-b6cb-b1abd2f3e4ef.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc4NTkwMjcsIm5iZiI6MTc2Nzg1ODcyNywicGF0aCI6Ii8xNzYzOTE0MjQvNTMzMjM4OTMzLWU2YjBlMzcxLTQyMzItNDUwNC1iNmNiLWIxYWJkMmYzZTRlZi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwOFQwNzUyMDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03NzNhYmU2ZGU0NGQ0OTEwZjdlMGFkMTIzNzc5NWZmMjBjYjEyZmU0YzMxNmM5YjdlZmU1NWU2ZWRiNTQzNTdjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.4m6vL_AXDyUy4sPf4UzrqyhOSbkd6eDfrTkOTU3-KfA" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="20%"><video src="https://private-user-images.githubusercontent.com/176391424/530970789-e0741e1a-50b8-4aa1-9a72-e516c6081ca0.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc4NTkyMDgsIm5iZiI6MTc2Nzg1ODkwOCwicGF0aCI6Ii8xNzYzOTE0MjQvNTMwOTcwNzg5LWUwNzQxZTFhLTUwYjgtNGFhMS05YTcyLWU1MTZjNjA4MWNhMC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwOFQwNzU1MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iZGMzZmYwNjlkMzc4Y2JjYTliZmNkZjM3OGVjNmM0ODM1NDRlNDZlYTcwZDMxNGI2NWRiOTkwY2IyODM3NDFkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.5vf3LgFp6s2AxaTXFkCH7ucfJePEED_wrfRknh2ss3k" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="20%"><video src="https://private-user-images.githubusercontent.com/176391424/533239130-b64af207-2478-4702-88a8-43096ed6e426.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc4NTkxMTksIm5iZiI6MTc2Nzg1ODgxOSwicGF0aCI6Ii8xNzYzOTE0MjQvNTMzMjM5MTMwLWI2NGFmMjA3LTI0NzgtNDcwMi04OGE4LTQzMDk2ZWQ2ZTQyNi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwOFQwNzUzMzlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yODc5NWIyZDBhYmQ1YzA2ZjQ4Mjc3NzY5NTcxOGFjNDE1ODQ3ODQ0ZWNjN2JhOWNiMDQwMDEwY2IyZmQ3YWY3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.prPQCf_crC1R5XRJbDMya0rmDLoB8bJUkRkTF5ftOjA" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="20%"><video src="https://private-user-images.githubusercontent.com/176391424/533246354-ae980242-d541-4c80-bcad-0950ec6ee8bc.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc4NjAxNDEsIm5iZiI6MTc2Nzg1OTg0MSwicGF0aCI6Ii8xNzYzOTE0MjQvNTMzMjQ2MzU0LWFlOTgwMjQyLWQ1NDEtNGM4MC1iY2FkLTA5NTBlYzZlZThiYy5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwOFQwODEwNDFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMWE0MTUwODEzYWIyYTNhMTBjNTg2NTRjYTA0NDQ4ZDdjYWY5ZDk5YWIzMTQ4YTliNDBiNWIyYWRjN2YzYTRjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.3k5lDv8lYsZzuI0uMuqv7buKh2wCNwvB7QrPvbih-QE" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
      <td width="20%"><video src="https://private-user-images.githubusercontent.com/176391424/533239776-e1eaa8b3-3023-4796-a873-f1d8994db7bb.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc4NTkxNjEsIm5iZiI6MTc2Nzg1ODg2MSwicGF0aCI6Ii8xNzYzOTE0MjQvNTMzMjM5Nzc2LWUxZWFhOGIzLTMwMjMtNDc5Ni1hODczLWYxZDg5OTRkYjdiYi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwOFQwNzU0MjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMTM4ZGRlNmFlYTgzOGE1NmUzNDgyNTdmNTAyYmJlN2E5ZjEwYTVmYzA4ZDViYWJhMzMxNzUzZTc5ODgyNzExJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.MhO04P2MC9PY4iTkqsaGiGEgRR42sBLyEC8S-AXp_ic" style="width:100%; aspect-ratio:448/832; object-fit:cover;" controls loop></video></td>
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
pip install flash_attn==2.8.1 --no-build-isolation
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
conda install -c conda-forge ffmpeg
```
### 🤗 Model download
| Model Component | Description | Link |
| :--- | :--- | :---: |
| `chinese-wav2vec2-base` | chinese-wav2vec2-base | 🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |
| `SoulX-FlashTalk-14B` | Our 14b model| 🤗 [Huggingface](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B) |

```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashTalk-14B --local-dir ./models/SoulX-FlashTalk-14B
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./models/chinese-wav2vec2-base
```
### 🚀 Inference
```bash
# infer on single GPU
bash inference_script_single_gpu.sh
# infer on multy GPUs
bash inference_script_multi_gpu.sh
```

### 👋 Online Demo 
Coming Soon!


## 📧 Contact Us
If you are interested in leaving a message to our work, feel free to email le.shen@mail.dhu.edu.cn or qiaoqian@soulapp.cn or yutan@soulapp.cn or zhouke@soulapp.cn or liusiyuan@soulapp.cn

You’re welcome to join our WeChat group for technical discussions, updates.
<p align="center">
  <!-- <em>Due to group limits, if you can't scan the QR code, please add my WeChat for group access  -->
      <!-- : <strong>Tiamo James</strong></em> -->
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/wechat_group.png" width="300" alt="WeChat Group QR Code"/>
  </span>
  <!-- <span style="display: inline-block;">
    <img src="assets/wechat_tiamo.jpg" width="300" alt="WeChat QR Code"/>
  </span> -->
</p>

 
## 📚 Citation

If you find our work useful in your research, please consider citing:

```
@misc{shen2025soulxflashtalktechnicalreport,
      title={SoulX-FlashTalk: Real-Time Infinite Streaming of Audio-Driven Avatars via Self-Correcting Bidirectional Distillation}, 
      author={Le Shen and Qian Qiao and Tan Yu and Ke Zhou and Tianhang Yu and Yu Zhan and Zhenjie Wang and Ming Tao and Shunshun Yin and Siyuan Liu},
      year={2025},
      eprint={2512.23379},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.23379}, 
}
```

## 🙇 Acknowledgement
- [Infinitetalk](https://github.com/MeiGen-AI/InfiniteTalk) and [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon.
- [Self forcing](https://github.com/guandeh17/Self-Forcing): the codebase we built upon.
- [DMD](https://github.com/tianweiy/DMD2) and [Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus): the key distillation technique used by our method.
> [!TIP]
> If you find our work useful, please also consider starring the original repositories of these foundational methods.

## 💡 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Soul-AILab/SoulX-FlashTalk&type=date&legend=top-left)](https://www.star-history.com/#Soul-AILab/SoulX-FlashTalk&type=date&legend=top-left)