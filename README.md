# SFAT
This repository contains the codes and data of our paper: **Semantic Frame Aggregation-based Transformer for Live Video Comment Generation**.

## VideoChat Dataset
Processed Dataset can be downloaded from the [Google-Drive](https://drive.google.com/drive/folders/1CJHMAt-_uSTOydhYLrH-I0NuaW0vf2KV?usp=sharing) and placed in `../dataset` folder.

## SFAT Model
SFAT model checkpoints (pretraining and training) can be downloaded from the [Google-Drive](https://drive.google.com/drive/folders/1MSe3_0LYnE_-XVBmke87uiQVhEtJvx_J?usp=sharing) and placed in `../model_saves/avc_transformer` folder.

- **Pretraining**: In the first stage, the text encoder for context comments is pre-trained using a masked language modeling (MLM) task. Pretraining script:
  ```python
     python3 main.py -model avc -d livechat -b 32 -e 100 -lr 1e-4 -m pretrain
  ```
  The model from pretraining will be saved in the `../model_saves/avc_transformer` folder.
  
  The pre-trained model checkpoint can be downloaded from the above Google Drive: `python pretrain_final.pth`.
- **Training**: In the second stage, the entire model, including the video and context comment encoders, as well as the decoder, is trained. The objective in this phase is to predict the target comment from the multimodal inputs. Train the model using the following script: 
  ```python
     python3 main.py -model avc -d livechat -e 200 -lr 1e-4 -b 32 -m train -l pretrain_final.pth
  ```
  The model from training will be saved in the `../model_saves/avc_transformer` folder.
  
  The trained model checkpoint can be downloaded from the above Google Drive: `checkpoint_0.0001_199e.pth`.
  
- **Evaluation**: Load the train checkpoint and evaluate the model using the following script:
  ```python
     python3 main.py -model avc -d livechat -lr 1e-4 -b 32 -m eval -l checkpoint_0.0001_199e.pth
  ```
  ## Citation
