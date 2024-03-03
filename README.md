# TTTS_v2(WIP)

## V2 is built upon the VALL-E style GPT, and VQ-VAE is a context-aware structure. The rest is the same as the master version.


# Demo
Coming soon.

# Install
```
pip install -e .
```
# Training
Training the model including many steps.

### 1. Tokenizer training
Use the `ttts/prepare/bpe_all_text_to_one_file.py` to merge all text you have collected. To train the tokenizer, check the `ttts/gpt/voice_tokenizer` for more info.

### 2. VQVAE training
Use the `1_vad_asr_save_to_jsonl.py` and `2_save_mel_to_disk.py` to preprocess dataset.
Use the following instruction to train the model.
```
accelerate launch ttts/vqvae/train.py
```

### 3. GPT training
Use `3_save_vq_to_disk.py` to preprocess mel vq. Run
```
accelerate launch ttts/gpt/train.py
```
to train the model.

### 4. Vocoder training
You can choose any one vocoder of the following. You must choose one, due to the output reconstructed by the vqvae has low sound quality.
#### 4.2 Diffusion + vocos
I chose the pretrained vocos as the vocoder for this project for no special reson. You can swap to any other ones like univnet.

After change the config.json properly
```
accelerate launch ttts/diffusion/train.py
```

