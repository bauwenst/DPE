import torch
from torch import Tensor


def viterbiSum_forward(sentence: str, hard_boundary_after: list[int], vocab: set[str], loglikelihoods: Tensor) -> Tensor:  # Predictions is an N x K grid.
    """
    Note that
        grid[end] = sum_{all segmentations of string} P(segmentation)
                  = sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
                  = sum_{j=1..n} P(string[j:]) * grid[j]

    If the grid contains logs, and your probabilities are only available as logs, you want to convert the intermediate formula:
        grid[end] = log sum_{all segmentations of string} P(segmentation)
                  = log sum_{prefix of string ending at j=1..n} P(string[j:]) * sum_{all segmentations of string[:j]} P(segmentation)
                  = log sum_{j=1..n} exp(log(P(string[j:]))) * exp(log(sum_{all segmentations of string[:j]} P(segmentation)))
                  = log sum_{j=1..n} exp(log(P(string[j:])) + log(sum_{all segmentations of string[:j]} P(segmentation)))
                  = log sum_{j=1..n} exp(logP(string[j:end]) + grid[j])
    which is the backwards formula in He 2020.

    The forwards formula would be as follows: in node i, for j > i,
        grid[j] = log(  exp( grid[j] ) + exp( logP[i,j] + grid[i] )  )
    """
    n, max_k = loglikelihoods.shape
    assert n == len(sentence)

    log_marginal_probabilities = [0]*(n+1)  # Marginals for reaching the node in the segmentation graph before character i.
    for i in range(n):  # The starting index of the token.
        for k in range(min(n-i,max_k)):  # The length of the token.
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

    log_marginal_probabilities = [0]*(n+1)
    for i in range(n):  # The ending index of the token.
        log_marginal_probabilities[i+1] = torch.log(sum(torch.exp(   # Individual exp is very expensive.
            log_marginal_probabilities[i-k] + loglikelihoods[i-k][k]
        ) for k in range(min(max_k,i+1)) if sentence[i-k:i+1] not in vocab))  # This in-check can be precomputed by the DataLoader, just like the attention mask.

    return log_marginal_probabilities[-1]


def torch_rolls(input: Tensor, amounts: tuple[int,...], dim_roll: int, dim_vary: int) -> Tensor:
    """
    Version of torch.roll where you roll by a certain amount in one dimension, and the amount you roll varies with
    a second dimension.

    Based on the answers at https://stackoverflow.com/q/66596699/9352077
    """
    # Turn dims like -1 into their positive equivalent.
    dim_roll %= input.ndim
    dim_vary %= input.ndim

    dim_roll_size = input.shape[dim_roll]  # Call this R below
    dim_vary_size = input.shape[dim_vary]  # Call this V below
    assert dim_vary_size == len(amounts)

    broadcast_amounts = torch.tensor(amounts)      .view((1,)*dim_vary + (-1,) + (1,)*(input.ndim - dim_vary - 1))  # From [V] to [1 x ... x 1 x 1 x V x ... x 1]
    rolled_indices    = torch.arange(dim_roll_size).view((1,)*dim_roll + (-1,) + (1,)*(input.ndim - dim_roll - 1))  # From [R] to [1 x ... x 1 x R x 1 x ... x 1].
    rolled_indices = (rolled_indices - broadcast_amounts) % dim_roll_size  # Produces [1 x ... x 1 x R x V x ... x 1], i.e. V copies of the R indices, rolled differently each time.
    rolled_indices = torch.broadcast_to(rolled_indices, input.shape)

    return torch.gather(input, dim_roll, rolled_indices)


def batch_backward_viterbi_sum(forward_loglikelihoods: Tensor, forward_vocabulary_indices: Tensor) -> Tensor:
    """
    Some explanation here:
        - The log likelihoods are forwards. That is: at position n, you get the probability of stepping over any token
          to the RIGHT. These are thus reduced (using a tensor that tells you which of the |V| indices is relevant) to
          the probability of the at most K neighbouring tokens to your RIGHT in the graph.
        - Yet, in the formula, you need log likelihoods for your K neighbours to the LEFT. (Same holds for the in-vocabulary mask.)

    To convert from forwards to backwards format, for node (row) R in a forwards tensor, the backwards neighbours B
    live at a diagonal above it. For L=12, K=6:

        X X X X X X
        X X X X X B
        X X X X B X
        X X X B X X
        X X B X X X
        X B X X X X
        B X X X X X
        R R R R R .
        X X X X . .
        X X X . . .
        X X . . . .
        X . . . . .

    That is: you take the smallest step forward from your previous neighbour, or 2 from the neighbour before, or 3 from the neighbour before that, etc...
    So if you roll the first column down by 0, the second by 1, ..., you end up with

        X . . . . .
        X X . . . .
        X X X . . .
        X X X X . .
        X X X X X .
        X X X X X X
        B B B B B B
        R X X X X X
        X R X X X X
        X X R X X X
        X X X R X X
        X X X X R X

    In both representations, one of the L+1 graph nodes is not represented: the last node has no row in the forwards
    mask because it has no successors, but it does have a row in the backwards mask because it has predecessors);
    and similarly, the first node has a bunch of successors but no predecessors. So they are indexed differently.

    :param forward_loglikelihoods:     B x L x V
    :param forward_vocabulary_indices: B x L x K
    :return: single-number likelihood
    """
    B, L, K = forward_vocabulary_indices.shape

    # Input
    forward_loglikelihoods = torch.gather(forward_loglikelihoods, dim=2, index=forward_vocabulary_indices)  # B x L x K
    backward_loglikelihoods = torch_rolls(forward_loglikelihoods,          tuple(range(K)), dim_roll=1, dim_vary=2)
    backward_mask           = torch_rolls(forward_vocabulary_indices >= 0, tuple(range(K)), dim_roll=1, dim_vary=2)    # B x L x K
    # The first non-batch dimension of these tensors is still indexed forwards, except shifted by one (index 0 is the second node, i.e. the one after the first character).
    # The second dimension is backwards: e.g., in the last row, index 0 points to node L-1, index 1 to node L-2, ...

    # Output
    log_marginals = torch.zeros(B,L+1)  # B x (L+1)  This tensor stores the graph in reverse order: earlier computations are further to the right. This is exactly like the above inputs: larger steps are further to the right.
    for i in range(L-1, -1, -1):  # Iterates the grid in reverse order. Note that there are only L iterations (L-1, ..., 0), not L+1 (the size of the grid) because the grid has a dummy all the way on the right (index L).
        n_predecessors = min(K,L-i)
        log_marginals[:,i] = torch.logsumexp(log_marginals[:, i+1:i+1+n_predecessors] + backward_mask[:, L-1-i, :n_predecessors]*backward_loglikelihoods[:, L-1-i, :n_predecessors], dim=-1)

    return 1/B*torch.sum(log_marginals[:,0])
