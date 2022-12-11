ct = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
ct_prime = [1, 1, 0, 1, 0, 0, 1, 1]

def levenshtein(ct, ct_prime):
    if not ct: return len(ct_prime)
    if not ct_prime: return len(ct)
    return min(levenshtein(ct[1:], ct_prime[1:]) + (ct[0] != ct_prime[0]),
               levenshtein(ct[1:], ct_prime) + 1,
               levenshtein(ct, ct_prime[1:]) + 1)


print(levenshtein(ct, ct_prime))