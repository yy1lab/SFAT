# SFAT
Semantic Frame Aggregation-based Transformer for Live Video Comment Generation

## Dataset
Processed Dataset can be downloaded from the [Google-Drive](https://drive.google.com/drive/folders/1CJHMAt-_uSTOydhYLrH-I0NuaW0vf2KV?usp=sharing) and placed in ../dataset folder.

## SFAT Model
SFAT model checkpoints can be downloaded from the [Google-Drive](https://drive.google.com/drive/folders/1v3GpjvqF6t5vOEOdcTWLKVNlQ_D-1tNu?usp=sharing) and placed in ../model_saves/avc_transformer folder.

- Pretraining: In the first stage, the text encoder for context comments is pre-trained using a masked language modeling (MLM) task. The pretrained model checkpoint is available in the above folder: pretrain_final.pth.

- Training: Training: In the second stage, the entire model including the video and context comment encoders, as well as the decoder, is trained. The objective in this phase is to predict the target comment from the multimodal inputs. The trained model checkpoint is available in the above folder: checkpoint_0.0001_199e.pth.
  
- Evaluation:
