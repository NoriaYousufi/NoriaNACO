ct = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
ct_prime = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]

def hamming2(ct, ct_prime):
    """Calculate the Hamming distance between two bit strings"""
    assert len(ct) == len(ct_prime)
    return sum(c1 == c2 for c1, c2 in zip(ct, ct_prime))

print(hamming2(ct, ct_prime))

# assert hamming2("1010", "1111") == 2
# assert hamming2("1111", "0000") == 4
# assert hamming2("1111", "1111") == 0