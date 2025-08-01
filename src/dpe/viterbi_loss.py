import torch
from torch import Tensor
from torch.nn import LogSoftmax


def viterbiSum_forward(sentence: str, hard_boundary_after: list[int], vocab: set[str], loglikelihoods: Tensor) -> Tensor:  # Predictions is an N x K grid.
    n, max_k = loglikelihoods.shape
    assert n == len(sentence)

    log_marginal_probabilities = [0]*n  # Note that
    # grid[end] = sum_{all segmentations of string} P(segmentation)
    #           = sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
    #           = sum_{j=1..n} P(string[j:]) * grid[j]
    #
    # If the grid contains logs, and your probabilities are only available as logs, you want to convert the intermediate formula:
    # grid[end] = log sum_{all segmentations of string} P(segmentation)
    #           = log sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
    #           = log sum_{j=1..n} exp(log(P(string[j:]))) * exp(log(sum_{all segmentations of string[:j]} P(segmentation)))
    #           = log sum_{j=1..n} exp(log(P(string[j:])) + log(sum_{all segmentations of string[:j]} P(segmentation)))
    #           = log sum_{j=1..n} exp(logP(string[j:end]) + grid[j])
    # which is the backwards formula in He 2020.
    #
    # The forwards formula would be as follows: in node i, for j > i,
    #  grid[j] = log(  exp( grid[j] ) + exp( logP[i,j] + grid[i] )  )

    for i in range(n):  # The starting index of the token.
        for k in range(max_k):  # The length of the token.
            if hard_boundary_after and i+k >= hard_boundary_after[0]:  # TODO: Could be off-by-one error.
                hard_boundary_after.pop(0)
                break  # Longer tokens are no longer allowed by the next hard boundary.

            if sentence[i:i+k+1] not in vocab:
                continue

            log_marginal_probabilities[i+k+1] = torch.log( torch.exp(log_marginal_probabilities[i+k+1]) + torch.exp(log_marginal_probabilities[i] + loglikelihoods[i][k]) )

    return log_marginal_probabilities[-1]


def viterbiSum_backward(sentence: str, vocab: set[str], loglikelihoods: Tensor) -> Tensor:  # Predictions is an N x K grid.
    n, max_k = loglikelihoods.shape
    assert n == len(sentence)

    log_marginal_probabilities = [0]*n  # Note that
    # grid[end] = sum_{all segmentations of string} P(segmentation)
    #           = sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
    #           = sum_{j=1..n} P(string[j:]) * grid[j]
    #
    # If the grid contains logs, and your probabilities are only available as logs, you want to convert the intermediate formula:
    # grid[end] = log sum_{all segmentations of string} P(segmentation)
    #           = log sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
    #           = log sum_{j=1..n} exp(log(P(string[j:]))) * exp(log(sum_{all segmentations of string[:j]} P(segmentation)))
    #           = log sum_{j=1..n} exp(log(P(string[j:])) + log(sum_{all segmentations of string[:j]} P(segmentation)))
    #           = log sum_{j=1..n} exp(logP(string[j:end]) + grid[j])
    # which is the backwards formula in He 2020.
    for i in range(n):  # The starting index of the token.
        log_marginal_probabilities[i] = torch.log(sum(torch.exp(   # TODO: torch.logsumexp exists bro
            log_marginal_probabilities[i-k-1] + loglikelihoods[i-k-1][k]
        )
        for k in range(min(max_k,i)) if sentence[i-k-1:i] not in vocab))  # TODO: This in-check should be precomputed by the DataLoader, just like the attention mask. It should be implemented with masks.

    return log_marginal_probabilities[-1]
