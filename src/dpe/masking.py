

def matrixSegmentationDAG(sentence: str,  vocab: set[str], K: int, L: int=None) -> list[list[bool]]:  # L x K
    """
    Note: can be generated using the one-liner
        torch.tensor(vocabularyIndicesOfStrides) >= 0
    """
    # TODO: Preprocessing should be done whenever you compare to the vocabulary! There are no spaces in the vocab!
    L = L or len(sentence)
    mask = [[False]*K for _ in range(L)]
    L = len(sentence)
    for i in range(L):  # Character in sentence
        for k in range(min(K, L-i)):  # Token index from this character onward. Has been tested against off-by-one.
            if sentence[i:i+k+1] in vocab:
                mask[i][k] = True
    return mask


def batchMatrixSegmentationDAG(sentences: list[str], vocab: set[str], max_k: int=None) -> list[list[list[bool]]]:  # B x L x K
    max_L =          max(map(len, sentences))
    max_k = max_k or max(map(len, vocab))
    return [matrixSegmentationDAG(sentence, vocab, max_k, max_L) for sentence in sentences]


def vocabularyIndicesOfStrides(sentence: str, vocab: dict[str,int], K: int, L: int=None) -> list[list[int]]:
    """
    Returns an L x K matrix where for each character l, the kth value is the index of token s[l:l+k+1] in the vocabulary.
    """
    L = L or len(sentence)
    vocab_indices = [[-1]*K for _ in range(L)]
    L = len(sentence)
    for i in range(L):
        for k in range(min(K, L-i)):  # Has been checked to not suffer from off-by-one.
            token = sentence[i:i+k+1]
            if token in vocab:
                vocab_indices[i][k] = vocab[token]

    return vocab_indices


def batchVocabularyIndicesOfStrides(sentences: list[str], vocab: dict[str,int], max_k: int=None) -> list[list[list[int]]]:  # B x L x K
    max_L =          max(map(len, sentences))
    max_k = max_k or max(map(len, vocab))
    return [vocabularyIndicesOfStrides(sentence, vocab, max_k, max_L) for sentence in sentences]


if __name__ == "__main__":
    print(vocabularyIndicesOfStrides("this is an example sentence", {"hi": 69}, K=4))
