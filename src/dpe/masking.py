

def matrixSegmentationDAG(sentence: str,  vocab: set[str], K: int, L: int=None) -> list[list[bool]]:  # L x K
    # TODO: Possible you want pretokenisation here, but not entirely sure if so for training.
    L = L or len(sentence)
    mask = [[False]*K for _ in range(L)]
    for i in range(L):  # Character in sentence
        for k in range(min(K, L-i)):  # Token index from this character onward TODO: Check for off-by-one
            if sentence[i:i+k+1] in vocab:
                mask[i][k] = True
    return mask


def batchMatrixSegmentationDAG(sentences: list[str], vocab: set[str], max_k: int=None) -> list[list[list[bool]]]:  # B x L x K
    max_L =          max(map(len, sentences))
    max_k = max_k or max(map(len, vocab))
    return [matrixSegmentationDAG(sentence, vocab, max_k, max_L) for sentence in sentences]
