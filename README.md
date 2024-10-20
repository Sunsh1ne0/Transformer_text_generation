## Hand-made transformer model

### Setup
```
git clone https://github.com/Sunsh1ne0/Transformer_text_generation.git
cd Transformer_text_generation
export PYTHONPATH=$PYTHONPATH:$CWD
```
Install all the dependencies
### Training
```
python train/train.py # train model with classic positional encoding
python train/train_RoPE.py # train model with a rotary positional encoding
```
## Run
```
python text_generation.py
```
