# TTTS_v2(WIP)

## V2 is built upon the VALL-E style GPT, and VQ-VAE is a context-aware structure. The rest is the same as the master version.
This project is for training tortoise-tts like model.
Most of the codes are from [tortoise tts](https://github.com/neonbjb/tortoise-tts) and [xtts](https://github.com/coqui-ai/TTS/tree/dev/TTS/tts/layers/xtts).
The distinguishing factor lies in certain training details and the diffusion model. This repository employs the same architecture as animate-anyone, incorporating a referencenet for enhanced zero-shot performance.

![image](arch.png)

Now only support mandarin.
Pretrained model can be found in [here](https://huggingface.co/adelacvg/TTTS/tree/main), you can use colab to generate any speech.

# Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adelacvg/ttts/blob/master/demo.ipynb)

| refer | input | output |
| :----| :---- | :---- |
|[refer.webm](https://github.com/adelacvg/ttts/assets/27419496/a6cf9634-cf09-4e27-baf9-b4f86ce6533c)|四是四，十是十，十四是十四，四十是四十。| [gen0.webm](https://github.com/adelacvg/NS2VC/assets/27419496/3defcd4a-6843-464c-a903-285a14751096)|
|[refer.webm](https://github.com/adelacvg/ttts/assets/27419496/ade50b3b-5ab3-4a8a-b9e9-977cb9b19ea1)|八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮。|[out2.webm](https://github.com/adelacvg/ttts/assets/27419496/b40ba548-7cf5-4b73-8f8d-36a532b53848)|
|[refer.webm](https://github.com/adelacvg/ttts/assets/27419496/866d7222-734a-4a87-a6cd-e92bf71f8408)|黑化肥发灰，灰化肥发黑。黑化肥挥发会发灰，灰化肥挥发会发黑。|[out3.webm](https://github.com/adelacvg/ttts/assets/27419496/b5372a53-e3bc-418c-bfd3-3f2afb96e56d)|
# Install
```
pip install -e .
```
# Training
Training the model including many steps.

### 1. Tokenizer training
Use the `ttts/prepare/bpe_all_text_to_one_file.py` to merge all text you have collected. To train the tokenizer, check the `ttts/gpt/voice_tokenizer` for more info.

### 2. VQVAE training
Use the `vad_asr_save_to_jsonl.py` and `save_mel_to_disk.py` to preprocess dataset.
Use the following instruction to train the model.
```
accelerate launch ttts/vqvae/train.py
```

### 3. GPT training
Use `save_mel_vq_to_disk.py` to preprocess mel vq. Run
```
accelerate launch ttts/gpt/train.py
```
to train the model.

### 4. Vocoder training
You can choose any one vocoder of the following. You must choose one, due to the output reconstructed by the vqvae has low sound quality.
#### 4.1 Hifigan
WIP
#### 4.2 Diffusion + vocos
I chose the pretrained vocos as the vocoder for this project for no special reson. You can swap to any other ones like univnet.

After change the config.json properly
```
accelerate launch ttts/diffusion/train.py
```

