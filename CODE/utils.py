from argparse import ArgumentParser
from sys import argv
from sacrebleu.metrics import BLEU
from rouge import Rouge
import torch

def parse_args(args=argv[1:]):
    """
    Parse command-line arguments.

    Args:
        args (List[str], optional): Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = ArgumentParser(description="Train the model")
    parser.add_argument("-e", default=100, help="Number of epochs", type=int)
    parser.add_argument("-lr", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("-b", default=32, help="Batch size", type=int)
    parser.add_argument("-l", default="", help="Load model -> path to file", type=str)
    parser.add_argument("-m", default="train", help="Modes: [train, eval, gen, pretrain] - default=train", type=str)
    parser.add_argument("-model", default="avc", help="Model: [avc, av, ac, c] - default=avc", type=str)
    parser.add_argument("-d", default="livechat", help="Dataset: [livechat, gdialogue] - default=livechat", type=str)
    return parser.parse_args(args)

def bleu_score(generated_sample, original_sample):
    """
    DEPRECATED
    Calculate the BLEU score between a generated sample and an original sample.

    Args:
        generated_sample (str): The generated sample (hypothesis).
        original_sample (str): The original sample (reference).

    Returns:
        sacrebleu.BLEUScore: The BLEU score object.
    """
    bleu = BLEU()
    return bleu.sentence_score(hypothesis=generated_sample, reference=[original_sample])

def rouge_score(generated_sample, original_sample):
    """
    DEPRECATED
    Calculate the ROUGE scores between a generated sample and an original sample.

    Args:
        generated_sample (str): The generated sample (hypothesis).
        original_sample (str): The original sample (reference).

    Returns:
        dict: A dictionary containing ROUGE scores.
    """
    rouge = Rouge()
    return rouge.get_scores(hyps=generated_sample, refs=original_sample)

def recall(hit_rank: torch.tensor, k=1):
    """
    Calculate recall@k.

    Args:
        hit_rank (torch.tensor): A tensor containing the ranks of correct items (0-based indexing).
        k (int, optional): The value of k for recall@k. Defaults to 1.

    Returns:
        float: Recall@k in percentage.
    """
    batch_size=hit_rank.size(0)
    num_lower_ranks = (hit_rank < k).sum().item() # all ranks below k for given batch
    return num_lower_ranks*100/batch_size

def mean_rank(hit_rank):
    """
    Calculate the mean rank.

    Args:
        hit_rank (torch.tensor): A tensor containing the ranks of correct items (0-based indexing).

    Returns:
        float: The mean rank.
    """
    mean_rank = (hit_rank+1).float().mean().item()
    return mean_rank

def mean_reciprocal_rank(hit_rank):
    """
    Calculate the mean reciprocal rank.

    Args:
        hit_rank (torch.tensor): A tensor containing the ranks of correct items (0-based indexing).

    Returns:
        float: The mean reciprocal rank.
    """
    reciprocal_ranks = 1.0 / (hit_rank+1).float()
    mean_reciprocal_rank = reciprocal_ranks.mean().item()
    return mean_reciprocal_rank

