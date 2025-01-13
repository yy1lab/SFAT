import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.transformer_dropAudio import TransformerDecoder, TransformerDecoderLayer, VCDecoderLayer
from transformers import AutoModel, BertConfig, BertModel, BertForMaskedLM
from transformers import CLIPModel


class ContextEncoder(nn.Module):
    """
    ContextEncoder module for encoding context information using a pre-trained BERT model.

    Args:
        embedding_size (int): Size of the embedding used by the BERT model.
        hidden_size (int): Size of the hidden state used by the BERT model.
        num_layers (int, optional): Number of layers in the BERT model. Defaults to 2.
        p (float, optional): Dropout probability. Defaults to 0.1.
        batch_first (bool, optional): If True, the input is expected to be of shape (batch_size, seq_length, embedding_size).
            If False, the input is expected to be of shape (seq_length, batch_size, embedding_size). Defaults to True.
    """
    
    def __init__(
            self, 
            embedding_size, 
            hidden_size,
            num_layers=2, 
            p=0.1, 
            batch_first=True
        ):
        
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-mini")
        
        self.pretrain_prediction = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, x, x_attn_mask):
        """
        Forward pass of the ContextEncoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
            x_attn_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Encoded context tensor of shape (batch_size, seq_length, hidden_size).
        """
        
        model_output = self.bert(input_ids=x.view(x.size(0) * x.size(1), x.size(2)), attention_mask=x_attn_mask.view(x_attn_mask.size(0) * x_attn_mask.size(1), x_attn_mask.size(2)))
        bert_encoded = model_output.last_hidden_state
        bert_encoded = bert_encoded.view(x.size(0), x.size(1), x.size(2), self.hidden_size)
        bert_encoded = bert_encoded[:, :, 0, :]
        
        return bert_encoded
    
    def pretrain(self, x, x_attn_mask):
        """
        Pre-training method of the ContextEncoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
            x_attn_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, seq_length, vocab_size).
        """
        model_output = self.bert(input_ids=x.view(x.size(0) * x.size(1), x.size(2)), attention_mask=x_attn_mask.view(x_attn_mask.size(0) * x_attn_mask.size(1), x_attn_mask.size(2)))
        bert_encoded = model_output.last_hidden_state
        bert_encoded = bert_encoded.view(x.size(0), x.size(1), x.size(2), self.hidden_size)
        predictions = []
        for i in range(bert_encoded.size(1)):
            predictions.append(self.pretrain_prediction(bert_encoded[:, i, :, :]))
        predictions = torch.stack(predictions, dim=1)
        return predictions

# class TranscriptEncoder(nn.Module):
#     """
#     TranscriptEncoder module for encoding transcript information using a pre-trained BERT model.

#     Args:
#         embedding_size (int): Size of the embedding used by the BERT model.
#         hidden_size (int): Size of the hidden state used by the BERT model.
#         num_layers (int, optional): Number of layers in the BERT model. Defaults to 2.
#         p (float, optional): Dropout probability. Defaults to 0.1.
#         batch_first (bool, optional): If True, the input is expected to be of shape (batch_size, seq_length, embedding_size).
#             If False, the input is expected to be of shape (seq_length, batch_size, embedding_size). Defaults to True.
#     """
#     def __init__(
#             self, 
#             embedding_size, 
#             hidden_size, 
#             num_layers=2, 
#             p=0.1, 
#             batch_first=True
#         ):
#         super(TranscriptEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding_size = embedding_size
#         self.num_layers = num_layers

#         self.bert = AutoModel.from_pretrained("prajjwal1/bert-mini")

#         self.pretrain_prediction = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

#     def forward(self, x, x_attn_mask):
#         """
#         Forward pass of the TranscriptEncoder module.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
#             x_attn_mask (torch.Tensor): Aglobals()['set_trace'] = lambda:1ttention mask tensor of shape (batch_size, seq_length).

#         Returns:
#             torch.Tensor: Encoded transcript tensor of shape (batch_size, 1, hidden_size).
#         """
#         
#         bert_encoded = self.bert(x, x_attn_mask).last_hidden_state
#         bert_encoded = bert_encoded[:, 0, :].unsqueeze(1) #get 1st token for summary([3x256]), add a dimension(unsqueeze) to it([3x1x256]) to match that of context_enc [3x15x10]

#         return bert_encoded
    
#     def pretrain(self, x, x_attn_mask):
#         """
#         Pre-training method of the TranscriptEncoder module.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
#             x_attn_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length).

#         Returns:
#             torch.Tensor: Predicted tensor of shape (batch_size, vocab_size).
#         """
#         
        
#         bert_encoded = self.bert(x, x_attn_mask).last_hidden_state
#         prediction = self.pretrain_prediction(bert_encoded)
#         return prediction

class VideoEncoder(nn.Module):
    """
    VideoEncoder module for encoding video features using a transformer-based architecture.

    Args:
        features_size (int): Size of the input video features.
        hidden_size (int): Size of the hidden state used in the transformer.
        num_layers (int, optional): Number of transformer layers. Defaults to 2.
        p (float, optional): Dropout probability. Defaults to 0.1.
        batch_first (bool, optional): If True, the input is expected to be of shape (batch_size, seq_length, features_size).
            If False, the input is expected to be of shape (seq_length, batch_size, features_size). Defaults to True.
    """
    def __init__(
            self,
            features_size, 
            hidden_size, 
            num_layers=2, 
            p=0.1, 
            batch_first=True
        ):
        super(VideoEncoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.features_size = features_size
        self.num_layers = num_layers
       
        self.fc = nn.Linear(features_size, features_size)
        self.relu = nn.ReLU()
        #self.positional_embedding = PositionalEncoding(2048, p)
        #change:: for clip embeddings: 512
        self.positional_embedding = PositionalEncoding(512, p)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                features_size, 
                8, 
                #512,
                256,
                p, 
                self.relu, 
                batch_first=batch_first
            ), 
            num_layers
        )
        self.linear_transformer = nn.Linear(features_size, hidden_size)
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.sm = torch.nn.Softmax(dim = 1)
        self.temperature = torch.tensor(0.1)


    def forward(self, x,clip_context,clip_context_am):
        """
        Forward pass of the VideoEncoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, features_size).

        Returns:
            torch.Tensor: Encoded video tensor of shape (batch_size, seq_length, hidden_size).
        """
       
        
        #trans_output = self.linear_transformer(trans_output)
        
        #change:: wt frames tokenize using clip
        #trans_output = self.linear_transformer(trans_output)
        
    
        
        x = self.dropout(self.fc(x))

        pos_embedding = self.positional_embedding(x)
        trans_output = self.transformer(pos_embedding)
        
        #trans_output = self.linear_transformer(trans_output)
        
        breakpoint()
        
        clip_input = clip_context.reshape(clip_context.size(0) * clip_context.size(1), -1).contiguous()
        clip_input_am = clip_context_am.reshape(clip_context_am.size(0) * clip_context_am.size(1), -1).contiguous()
        clip_context_features = self.clip.get_text_features(clip_input, attention_mask=clip_input_am).view(clip_context.size(0), clip_context.size(1), -1)
        
        n_video_features = torch.nn.functional.normalize(self.linear_transformer(trans_output), p=2, dim=-1)
        n_text_features = torch.nn.functional.normalize(self.linear_transformer(clip_context_features), p=2, dim=-1)
        
        scores = torch.bmm(n_video_features, n_text_features.permute(0,2,1))
        scores = self.sm(scores/self.temperature)
        wt_video_features = torch.bmm(scores.permute(0,2,1), n_video_features)
        
        breakpoint()
        sum_video_feat = torch.sum(wt_video_features, dim = 1)#attention over similarity=wgted sum

        
        #trans_output = self.linear_transformer(trans_output)
        return sum_video_feat.unsqueeze(1)

class CommentDecoder(nn.Module):
    """
    CommentDecoder module for decoding comments using a transformer-based architecture.

    Args:
        embedding_size (int): Size of the input embedding for decoding.
        hidden_size (int): Size of the hidden state used in the transformer.
        output_size (int): Size of the output tensor, i.e., the vocabulary size for comments.
        n_head (int, optional): Number of attention heads in the transformer. Defaults to 8.
        dim_ff (int, optional): Size of the feedforward layer in the transformer. Defaults to 2048.
        num_layers (int, optional): Number of transformer layers. Defaults to 2.
        p (float, optional): Dropout probability. Defaults to 0.1.
        batch_first (bool, optional): If True, the input is expected to be of shape (batch_size, seq_length, embedding_size).
            If False, the input is expected to be of shape (seq_length, batch_size, embedding_size). Defaults to True.
    """
    def __init__(
        self, 
        embedding_size, 
        hidden_size, 
        output_size,
        n_head=8,
        dim_ff=2048,
        num_layers=2, 
        p=0.1, 
        batch_first=True
    ):
        super(CommentDecoder, self).__init__()
        
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.n_head = n_head
        self.dim_ff = dim_ff

        self.relu = nn.ReLU()
        self.positional_embedding = PositionalEncoding(hidden_size, p, 200)
        self.transformer = TransformerDecoder(
            TransformerDecoderLayer(
                embedding_size, 
                n_head, 
                dim_ff, 
                dropout=p, 
                activation=self.relu,
                batch_first=batch_first
            ),
            self.num_layers
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, embedding, hidden_comments, hidden_video, x_mask=None, tgt_attn_mask=None):
        """
        Forward pass of the CommentDecoder module.

        Args:
            embedding (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
            hidden_comments (torch.Tensor): Hidden tensor for comments from the ContextEncoder module.
            hidden_audio (torch.Tensor): Hidden tensor for audio from the AudioEncoder module.
            hidden_video (torch.Tensor): Hidden tensor for video from the VideoEncoder module.
            x_mask (torch.Tensor, optional): Attention mask for the input tensor. Defaults to None.
            tgt_attn_mask (torch.Tensor, optional): Attention mask for the target tensor. Defaults to None.

        Returns:
            torch.Tensor: Decoded comment tensor of shape (batch_size, seq_length, output_size).
        """
        
        if x_mask != None:
            outputs = self.transformer(embedding, hidden_comments, hidden_video, tgt_mask=x_mask, tgt_key_padding_mask=~tgt_attn_mask.bool())
        else:
            outputs = self.transformer(embedding, hidden_comments, hidden_video)
        
        breakpoint()
        #outputs = outputs.squeeze(1)
        breakpoint()
        predictions = self.fc(outputs)
        return predictions

class VCDecoder(nn.Module):
    """
    VCDecoder module for decoding the responses using a transformer-based architecture.

    Args:
        embedding_size (int): Size of the input embedding for decoding.
        hidden_size (int): Size of the hidden state used in the transformer.
        output_size (int): Size of the output tensor, i.e., the vocabulary size for responses.
        n_head (int, optional): Number of attention heads in the transformer. Defaults to 8.
        dim_ff (int, optional): Size of the feedforward layer in the transformer. Defaults to 2048.
        num_layers (int, optional): Number of transformer layers. Defaults to 2.
        p (float, optional): Dropout probability. Defaults to 0.1.
        batch_first (bool, optional): If True, the input is expected to be of shape (batch_size, seq_length, embedding_size).
            If False, the input is expected to be of shape (seq_length, batch_size, embedding_size). Defaults to True.
    """
    def __init__(
        self, 
        embedding_size, 
        hidden_size, 
        output_size,
        n_head=8,
        dim_ff=2048,
        num_layers=2, 
        p=0.1, 
        batch_first=True
    ):
        super(VCDecoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.n_head = n_head
        self.dim_ff = dim_ff

        self.relu = nn.ReLU()
        self.positional_embedding = PositionalEncoding(hidden_size, p, 200)
        self.transformer = TransformerDecoder(
            VCDecoderLayer(
                embedding_size, 
                n_head, 
                dim_ff, 
                dropout=p, 
                activation=self.relu,
                batch_first=batch_first
            ),
            self.num_layers
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, embedding, hidden_comments, hidden_video, x_mask=None, tgt_attn_mask=None):
        """
        Forward pass of the VCDecoder module.

        Args:
            embedding (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_size).
            hidden_comments (torch.Tensor): Hidden tensor for comments from the CommentEncoder module.
            hidden_video (torch.Tensor): Hidden tensor for video from the VideoEncoder module.
            x_mask (torch.Tensor, optional): Attention mask for the input tensor. Defaults to None.
            tgt_attn_mask (torch.Tensor, optional): Attention mask for the target tensor. Defaults to None.

        Returns:
            torch.Tensor: Decoded response tensor of shape (batch_size, seq_length, output_size).
        """
       
        if x_mask != None:
            outputs = self.transformer(embedding, hidden_comments, None, hidden_video, tgt_mask=x_mask, tgt_key_padding_mask=~tgt_attn_mask.bool())
        else:
            outputs = self.transformer(embedding, hidden_comments, None, hidden_video)
        outputs = outputs.squeeze(1)
        predictions = self.fc(outputs)

        return predictions

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module for adding positional encoding to the input tensor.

    Args:
        d_model (int): The number of expected features in the input tensor.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        #sin(pos/10000^(2i/d_model)) for even and cos(pos/10000^(2i/d_model)) for odd
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Input tensor with positional encoding added, of shape (batch_size, seq_length, d_model).
        """
        x_reshaped = x.permute(1, 0, 2)
        x_reshaped = x_reshaped + self.pe[:x_reshaped.size(0)]
        return x_reshaped.permute(1, 0, 2)

class Discriminator(nn.Module):
    """
    OUTDATED!!
    Discriminator module for predicting the authenticity of the input samples.

    Args:
        hidden_size (int): The number of features in the hidden layer.
    """
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, modalities, target):
        """
        Forward pass of the discriminator.

        Args:
            modalities (torch.Tensor): Input tensor representing multiple modalities of shape (batch_size, num_modalities, modalities_features).
            target (torch.Tensor): Target tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: Output tensor containing the predicted authenticity scores, of shape (batch_size, 1).
        """
        combined = torch.cat((modalities.view(modalities.size(0), modalities.size(1)*modalities.size(2)), target.squeeze(1)), dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        
        return output
