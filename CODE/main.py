import os
import torch
from transformers import logging, AutoTokenizer
from livechat_wgtframes import StreamChatDataset
from models.avc_generative_wgtframes import AVCGenerative

import train.trainer_wgtframes as trainer
from utils import parse_args

import os
os.environ["PYTHONBREAKPOINT"] = "0"

if __name__=="__main__":
    """
    Main script for training generative dialogue models.

    This script initializes the necessary components for training a generative dialogue model and starts the training process.

    It sets up hyperparameters, loads the specified dataset, initializes the model, and starts the training using the Trainer class.

    Example:
        To train an AVCGenerative model on the 'livechat' dataset:
        python main.py -model avc -d livechat -e 100 -lr 1e-5 -b 32 -l model_save_name.pth -m train
    """
    args=parse_args()
    
    # Setting up hyperparameters
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    breakpoint();
    
    num_epochs = args.e
    learning_rate = args.lr
    batch_size = args.b
    filename_model = args.l
    mode = args.m
    dataset = args.d
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    input_size = 30522
    embedding_size = 256
    #embedding_size = 768
    hidden_size = 256
    #hidden_size = 768
    output_size = 30522
    num_layers_encoder = 4
    num_layers_decoder = 4
    #num_layers_encoder = 2
    #num_layers_decoder = 2
    enc_dropout = 0.1
    dec_dropout = 0.1
    weight_decay = 0.01
    comments_padding = 10
    transcript_padding = 100
    candidates_padding = 5
    nb_context_comments = 5
    nb_context_comments_eval = 15
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
    
    # Loading the dataset
    if dataset=="livechat":
        train_file = "train_reduced.json"
        test_file = "test_reduced.json"
        eval_file = "test_reduced_candidates.json"

        train_dataset = StreamChatDataset(
            tokenizer, 
            ".../dataset/", 
            "features_clip/", 
            train_file, 
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            nb_context_comments=nb_context_comments,
        )
        test_dataset = StreamChatDataset(
            tokenizer, 
            ".../dataset/", 
            "features_clip/", 
            test_file, 
            mode="train",
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            nb_context_comments=nb_context_comments
        )
        eval_dataset = StreamChatDataset(
            tokenizer, 
            ".../dataset/",
            "features_clip/", 
            eval_file, 
            mode="eval",
            comments_padding=comments_padding,
            transcript_padding=transcript_padding,
            candidates_padding=candidates_padding,
            nb_context_comments=nb_context_comments_eval
        )
    """ elif dataset=="gdialogue":
        train_file = "train_word.json"
        test_file = "val_word.json"
        eval_file = "test_word.json"
        
        train_dataset = GameBased(
            tokenizer, 
            ".../gamebased_clip",
            train_file,
            "train_clip_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="train"
        )
        test_dataset = GameBased(
            tokenizer,
            ".../gamebased_clip", 
            test_file,
            "val_clip_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="train"
        )
        eval_dataset = GameBased(
            tokenizer, 
            ".../gamebased_clip", 
            eval_file,
            "test_clip_feat.h5",
            comments_padding=comments_padding,
            nb_context_comments=nb_context_comments,
            mode="eval"
        ) """
        
       
    breakpoint();
    
    # Loading the model
    if args.model=="avc":
        breakpoint();
        
        model = AVCGenerative(
            input_size=input_size,
            output_size=output_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layer_encoder=num_layers_encoder,
            num_layer_decoder=num_layers_decoder,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            batch_first=True
        )
        
        save_dir=".../model_saves/avc_transformer"


        print("dir:",save_dir)

    """  elif args.model == "vc":
        model = VCGenerative(
            input_size=input_size,
            output_size=output_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layer_encoder=num_layers_encoder,
            num_layer_decoder=num_layers_decoder,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            batch_first=True
        )

        save_dir=".../model_saves/game_based10"
        print("dir:",save_dir)
         """
    

    trainer.start(
        model,
        num_epochs,
        learning_rate,
        batch_size,
        filename_model,
        device,
        mode,
        num_workers,
        tokenizer,
        train_dataset,
        test_dataset,
        eval_dataset,
        save_dir
    )
