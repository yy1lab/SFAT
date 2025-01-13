import random
import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset

from transformers import CLIPTokenizer

class StreamChatDataset(Dataset):
    """
    Custom PyTorch Dataset class for working with the StreamChat dataset.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer to encode chat text and other inputs.
        root (str, optional): Root directory path for the dataset. Defaults to "/media/livechat/dataset/".
        video_feature_dir (str, optional): Directory path for video features. Defaults to "features/".
        dataset_json (str, optional): JSON file containing dataset information. Defaults to "train.json".
        comments_padding (int, optional): Maximum length of chat comments after padding. Defaults to 50.
        transcript_padding (int, optional): Maximum length of audio transcripts after padding. Defaults to 100.
        candidates_padding (int, optional): Maximum length of candidate responses after padding (for evaluation mode). Defaults to 5.
        nb_context_comments (int, optional): Number of context comments to include in the chat context. Defaults to 30.
        mode (str, optional): Mode of operation ('train', 'eval', or 'gen'). Defaults to "train".
        allow_special_token (bool, optional): Whether to add special tokens during tokenization. Defaults to True.

    Attributes:
        root (str): Root directory path for the dataset.
        video_feature_dir (str): Directory path for video features.
        ds_json (list): Loaded JSON data from the dataset file.
        categories (list): List of video categories.
        id_video (list): List of video IDs.
        start (list): List of video start times.
        offsets (list): List of video offset start times.
        chat_context (list): List of chat contexts.
        resonse_indexes (list): List of response indexes.
        responses (list): List of response messages.
        transcript_audio (list): List of audio transcripts.
        mode (str): Mode of operation ('train', 'eval', or 'gen').
        tokenizer (transformers.AutoTokenizer): Tokenizer used for encoding text sequences.
        comments_padding (int): Maximum length of chat comments after padding.
        transcript_padding (int): Maximum length of audio transcripts after padding.
        candidates_padding (int): Maximum length of candidate responses after padding (for evaluation mode).
        nb_context_comments (int): Number of context comments to include in the chat context.
        allow_special_token (bool): Whether to add special tokens during tokenization.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a single sample from the dataset based on the given index.

    Note:
        For evaluation mode ('eval'), the dataset must contain a 'candidates' field in the JSON data.

    Example:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        livechat = StreamChatDataset(
            tokenizer,
            mode="eval",
            dataset_json="test_with_candidates.json"
        )
        print(len(livechat))  # Print the number of samples in the dataset.
        print(livechat[0])  # Print the first sample in the dataset.
    """
    def __init__(
            self, 
            tokenizer, 
            root="/media/yylab/STREAMCHAT/dataset/",
            video_feature_dir="features/",
            dataset_json="train.json",
            comments_padding=50,
            transcript_padding=100,
            candidates_padding = 5,
            nb_context_comments=30,
            mode="train",
            allow_special_token=True
        ):
        
        self.root = root
        self.video_feature_dir = video_feature_dir
        with open(os.path.join(root, dataset_json), "r") as data:
            self.ds_json = json.load(data)
        #self.ds_json = self.ds_json[:3]

        self.categories = [video["category"] for video in self.ds_json]
        self.id_video = [video["id_video"] for video in self.ds_json]
        self.start = [video["start"] for video in self.ds_json]
        self.offsets = [video["offset_start"] for video in self.ds_json]
        self.chat_context = [video["chat_context"] for video in self.ds_json]
        self.resonse_indexes = [video["response_index"] for video in self.ds_json]
        self.responses = [video["responses"] for video in self.ds_json]
        self.transcript_audio = [video["transcript_audio"] for video in self.ds_json]
        
        self.mode = mode
        if self.mode=="eval":
            self.candidates = [video["candidates"] for video in self.ds_json]
        self.tokenizer = tokenizer
        self.comments_padding = comments_padding
        self.transcript_padding = transcript_padding
        self.candidates_padding = candidates_padding
        self.nb_context_comments = nb_context_comments
        self.allow_special_token = allow_special_token
        
        self.tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            
    def __len__(self):
        return len(self.id_video)

    def __getitem__(self, index):
       
        chat_context = self.chat_context[index]
        responses = self.responses[index]
        transcript_audio = self.transcript_audio[index]

        chat_context_ids = []
        chat_context_am = []
        combined_context_tokens = []
        combined_context_am = []
        for _ in range(self.nb_context_comments):
            random_index = random.randint(0, len(chat_context)-1)
            chat_context_input = self.tokenizer.encode_plus(chat_context[random_index], add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
            chat_context_ids.append(chat_context_input["input_ids"])
            chat_context_am.append(chat_context_input["attention_mask"])
            #change:: wt frames tokenize using clip
            #breakpoint()
            #print(chat_context[random_index])
            chat_context_clip=self.tokenizer_clip(chat_context[random_index], add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
            #print(chat_context_clip.shape)    
            combined_context_tokens.append(chat_context_clip["input_ids"])
            combined_context_am.append(chat_context_clip["attention_mask"])
        
        #clip.tokenize(combined_context, max_length=200, padding='max_length', truncate=True)
        
        nb_responses = len(responses)
        resp_idx = random.randint(0, nb_responses-1)
        response_input = self.tokenizer.encode_plus(responses[resp_idx], add_special_tokens=self.allow_special_token, max_length=self.comments_padding, padding='max_length', truncation=True)
        response_ids = response_input["input_ids"]
        response_am = response_input["attention_mask"]

        transcript_audio_input = self.tokenizer.encode_plus(transcript_audio, add_special_tokens=self.allow_special_token, max_length=self.transcript_padding, padding='max_length', truncation=True)
        transcript_audio_input_ids = transcript_audio_input["input_ids"]
        transcript_audio_input_am = transcript_audio_input["attention_mask"]


        
        video_features = self.__getvideofeatures__(index)
        category = int(self.categories[index])

        candidates_ids, candidates_am = self.__getcandidates__(index) if self.mode=="eval" else ([0], [0])

        video_id=self.id_video[index]
         
        return {
            "chat_context": torch.tensor(chat_context_ids, dtype=torch.long),
            "chat_context_am": torch.tensor(chat_context_am, dtype=torch.long),

            "response": torch.tensor(response_ids, dtype=torch.long),
            "response_am": torch.tensor(response_am, dtype=torch.long),

            "transcript_audio": torch.tensor(transcript_audio_input_ids, dtype=torch.long),
            "transcript_audio_am": torch.tensor(transcript_audio_input_am, dtype=torch.long),

            "video_features": torch.tensor(video_features, dtype=torch.float),
            "category": torch.tensor(category, dtype=torch.long),
                        
            "candidates": torch.tensor(candidates_ids, dtype=torch.long),
            "candidates_am": torch.tensor(candidates_am, dtype=torch.long),
            
            "video_id": video_id,
            
            #change:: wt frames tokenize using clip
            "clip_context": torch.tensor(combined_context_tokens, dtype=torch.long),
            "clip_context_am": torch.tensor(combined_context_am, dtype=torch.long)
        }
    
    def __getvideofeatures__(self, index):
        
        video_features_file = self.root+self.video_feature_dir+self.categories[index]+\
            "/"+self.id_video[index]+"_"+str(self.start[index])+"_"+str(self.offsets[index])+".npy"
        video_features = np.load(video_features_file)
        return video_features
    
    def __getcandidates__(self, index):
        assert self.mode=="eval"

        candidates = self.candidates[index]
        candidates_ids = []
        candidates_am = []
        for candidate in candidates:
            candidate_input = self.tokenizer.encode_plus(candidate, add_special_tokens=self.allow_special_token, max_length=self.candidates_padding, padding='max_length', truncation=True)
            candidates_ids.append(candidate_input["input_ids"])
            candidates_am.append(candidate_input["attention_mask"])
        
        real_comment = candidates_ids[:1]
        real_comment_am = candidates_am[:1]
        
        plausible_comments = candidates_ids[1:10]
        plausible_comments_am = candidates_am[1:10]
        
        popular_comments = candidates_ids[21:30]
        popular_comments_am = candidates_am[21:30]
        
        random_comments = candidates_ids[61:70]
        random_comments_am = candidates_am[61:70]
        
        return real_comment+plausible_comments, real_comment_am+plausible_comments_am
        #return real_comment+popular_comments, real_comment_am+popular_comments_am
        #return real_comment+random_comments, real_comment_am+random_comments_am
       
if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    livechat = StreamChatDataset(
        tokenizer,
        mode="eval",
        dataset_json="test_with_candidates.json"
    )
    print(len(livechat))
    print(livechat[0])
