"""ComfyUI API graphs (Kijai WanVideoWrapper), taken from nina-hybrid/tools/comfy_job.py and vsr.py.

  wan22(...) — silent action clip, Wan 2.2 I2V A14B GGUF Q8 + lightx2v 4-step LoRAs (2 + 2 steps);
               121 frames -> RIFE x2 -> saved at 48 fps = real-time speed (graph from nina-gorokh/tools/comfy_job.py)
  talk(...)  — talking head, InfiniteTalk I2V on Wan 2.1 GGUF Q8 + lightx2v, 4 steps
Both end with an optional RTX VSR upscale (KJNodes ImageResizeKJv2 / nvidia_rtx_vsr) inside the same graph,
so a clip comes out of the pod already at the final size.
"""

NEG = ("bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
       "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
       "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
       "fused fingers, still picture, messy background, many people in the background, walking backwards")
STYLE = (" Static camera, no camera movement. Keep exactly the same framing and composition as the first frame, "
         "the camera stays on the same subject, no zoom out, steady handheld smartphone footage, natural daylight.")
TALK_PROMPT = ("an elderly grandmother in a headscarf talks warmly to the camera in her kitchen, "
               "natural head movement, friendly gestures")


def _common(image, W, H):
    return {
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": "wanvideo/Wan2_1_VAE_bf16.safetensors", "precision": "bf16"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "resize": {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["img", 0], "width": W, "height": H, "upscale_method": "lanczos",
            "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center",
            "divisible_by": 16, "device": "cpu"}},
        "size": {"class_type": "GetImageSizeAndCount", "inputs": {"image": ["resize", 0]}},
    }


def _finish(p, frames_src, prefix, fps, upscale, audio=None):
    images = frames_src
    if upscale:
        p["vsr"] = {"class_type": "ImageResizeKJv2", "inputs": {
            "image": frames_src, "width": upscale[0], "height": upscale[1], "upscale_method": "nvidia_rtx_vsr",
            "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center",
            "divisible_by": 16, "device": "gpu"}}
        images = ["vsr", 0]
    p["out"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": images, "frame_rate": float(fps), "loop_count": 0, "filename_prefix": prefix,
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False,
        "trim_to_audio": audio is not None, "pingpong": False, "save_output": True}}
    if audio is not None:
        p["out"]["inputs"]["audio"] = audio
    return p


def wan22(image, prompt, prefix, W=480, H=832, frames=121, seed=42, style=STYLE, upscale=(720, 1280),
          interp=2, fps=48):
    """interp: RIFE multiplier (1 = off). Wan 2.2 at 16 fps plays slow-mo; 121 frames x2 at 48 fps = real time.
    The old nina-hybrid clips were frames=81, interp=1, fps=16."""
    p = _common(image, W, H)
    lora = "WanVideo/Wan22/Wan_2_2_I2V_A14B_{}_lightx2v_4step_lora_260412_rank_64_fp16.safetensors"
    for tag, f in (("hi", "HighNoise"), ("lo", "LowNoise")):
        p[f"m_{tag}"] = {"class_type": "WanVideoModelLoader", "inputs": {
            "model": f"WanVideo/Wan22/Wan2.2-I2V-A14B-{f}-Q8_0.gguf", "base_precision": "fp16_fast",
            "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sageattn"}}
        p[f"l_{tag}"] = {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": lora.format("HIGH" if tag == "hi" else "LOW"), "strength": 1.0,
            "low_mem_load": False, "merge_loras": False}}
        p[f"ml_{tag}"] = {"class_type": "WanVideoSetLoRAs", "inputs": {"model": [f"m_{tag}", 0], "lora": [f"l_{tag}", 0]}}
    p.update({
        "t5": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {
            "model_name": "umt5-xxl-enc-fp8_e4m3fn.safetensors", "precision": "bf16",
            "load_device": "offload_device", "quantization": "fp8_e4m3fn"}},
        "text": {"class_type": "WanVideoTextEncode", "inputs": {
            "positive_prompt": prompt + (style or ""), "negative_prompt": NEG, "t5": ["t5", 0],
            "force_offload": True, "use_disk_cache": False, "device": "gpu"}},
        "i2v": {"class_type": "WanVideoImageToVideoEncode", "inputs": {
            "width": ["size", 1], "height": ["size", 2], "num_frames": frames, "noise_aug_strength": 0.0,
            "start_latent_strength": 1.0, "end_latent_strength": 1.0, "force_offload": True,
            "vae": ["vae", 0], "start_image": ["size", 0], "fun_or_fl2v_model": False, "tiled_vae": False}},
    })
    s = {"image_embeds": ["i2v", 0], "text_embeds": ["text", 0], "steps": 4, "cfg": 1.0, "shift": 5.0,
         "seed": seed, "force_offload": True, "scheduler": "euler", "riflex_freq_index": 0,
         "denoise_strength": 1.0, "batched_cfg": False, "rope_function": "comfy", "add_noise_to_samples": False}
    p["s_hi"] = {"class_type": "WanVideoSampler", "inputs": {**s, "model": ["ml_hi", 0], "start_step": 0, "end_step": 2}}
    p["s_lo"] = {"class_type": "WanVideoSampler", "inputs": {**s, "model": ["ml_lo", 0], "samples": ["s_hi", 0],
                                                             "start_step": 2, "end_step": -1}}
    p["dec"] = {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["vae", 0], "samples": ["s_lo", 0], "enable_vae_tiling": False,
        "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128, "normalization": "default"}}
    src = ["dec", 0]
    if interp > 1:
        p["rife"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": "rife49.pth", "frames": src, "clear_cache_after_n_frames": 10, "multiplier": interp,
            "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float32", "torch_compile": False,
            "batch_size": 1}}
        src = ["rife", 0]
    return _finish(p, src, prefix, fps, upscale)


def talk(image, audio, prefix, W=480, H=832, prompt=TALK_PROMPT, seed=2, blocks_to_swap=0, upscale=(720, 1280)):
    """blocks_to_swap: 0 on 40+ GB cards (peak 27 GB on A40), 20 on 24 GB cards (peak 17 GB on 3090)."""
    p = _common(image, W, H)
    model_in = {
        "model": "WanVideo/wan2.1-i2v-14b-480p-Q8_0.gguf", "base_precision": "fp16_fast",
        "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sageattn",
        "lora": ["lora", 0], "multitalk_model": ["itmodel", 0]}
    if blocks_to_swap:
        p["blockswap"] = {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": blocks_to_swap, "offload_img_emb": False, "offload_txt_emb": False,
            "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1, "block_swap_debug": False}}
        model_in["block_swap_args"] = ["blockswap", 0]
    p.update({
        "lora": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": "WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "strength": 1.0, "low_mem_load": False, "merge_loras": False}},
        "itmodel": {"class_type": "MultiTalkModelLoader", "inputs": {
            "model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"}},
        "model": {"class_type": "WanVideoModelLoader", "inputs": model_in},
        "clipv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "clipenc": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "clip_vision": ["clipv", 0], "image_1": ["size", 0], "strength_1": 1.0, "strength_2": 1.0,
            "crop": "center", "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}},
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "w2v": {"class_type": "Wav2VecModelLoader", "inputs": {
            "model": "wav2vec2-chinese-base_fp16.safetensors", "base_precision": "fp16", "load_device": "main_device"}},
        "w2vemb": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
            "wav2vec_model": ["w2v", 0], "audio_1": ["audio", 0], "normalize_loudness": True,
            "num_frames": 1000, "fps": 25.0, "audio_scale": 1.0, "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "i2v": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
            "vae": ["vae", 0], "width": ["size", 1], "height": ["size", 2], "frame_window_size": 81,
            "motion_frame": 9, "force_offload": False, "colormatch": "disabled", "start_image": ["size", 0],
            "tiled_vae": False, "clip_embeds": ["clipenc", 0], "mode": "infinitetalk", "output_path": ""}},
        "text": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": "umt5-xxl-enc-fp8_e4m3fn.safetensors", "precision": "bf16",
            "positive_prompt": prompt, "negative_prompt": NEG, "quantization": "fp8_e4m3fn",
            "use_disk_cache": True, "device": "gpu"}},
        "sampler": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["model", 0], "image_embeds": ["i2v", 0], "text_embeds": ["text", 0],
            "multitalk_embeds": ["w2vemb", 0], "steps": 4, "cfg": 1.0, "shift": 11.0, "seed": seed,
            "force_offload": True, "scheduler": "dpm++_sde", "riflex_freq_index": 0, "denoise_strength": 1.0,
            "batched_cfg": False, "rope_function": "comfy", "start_step": 0, "end_step": -1,
            "add_noise_to_samples": True}},
        "frames": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["sampler", 0]}},
    })
    return _finish(p, ["frames", 0], prefix, 25, upscale, audio=["audio", 0])
