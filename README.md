# TTTS_v2(WIP)

## V2 is built upon the VALL-E style GPT, and VQ-VAE-GAN is based on HierachySpeech++ and GPT-SoVITS.

# Demo
Coming soon.

# Install
```
pip install -e .
```
# Training
Training the model including two steps.

### 1. Tokenizer training
Use the `ttts/prepare/bpe_all_text_to_one_file.py` to merge all text you have collected. To train the tokenizer, check the `ttts/gpt/voice_tokenizer` for more info.

### 2. VQVAE training
Use the `1_vad_asr_save_to_jsonl.py` to preprocess dataset.
Use the following instruction to train the model.
```
python ttts/vqvae/train.py
```

### 3. GPT training
Use `2_save_vq_to_disk.py` to preprocess vq. Run
```
accelerate launch ttts/gpt/train.py
```
to train the model.

