CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python generate_video.py \
    --ckpt_dir models/SoulX-FlashTalk-14B \
    --wav2vec_dir models/chinese-wav2vec2-base \
    --input_prompt "A person is talking. Only the foreground characters are moving, the background remains static." \
    --cond_image examples/man.png \
    --audio_path examples/cantonese_16k.wav \
    --audio_encode_mode stream