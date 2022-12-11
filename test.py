ct =       [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
ct_prime = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]

best_score = -1
current_score = 0
check_point = 0
sequence_len = len(ct)

while check_point<sequence_len: 
    for i in range(check_point, sequence_len):
        if ct[i] == ct_prime[i]:
            current_score+=1
        if ct[i] != ct_prime[i]:
            break
    check_point = i + 1
    if current_score>best_score:
        best_score = current_score
        current_score = 0

print(best_score)
