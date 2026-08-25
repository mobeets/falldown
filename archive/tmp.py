#%%

K = 12

# generate sequences of holes in 4 layers, where hole count is 1-2-2-1
# and then find the sequences such that the best initial choice for greedy and 1-step is different than the best initial choice for 2-step

from itertools import combinations
sequences = set()
for holes1 in combinations(range(K), 1):
    for holes2 in combinations(range(K), 2):
        for holes3 in combinations(range(K), 2):
            for holes4 in combinations(range(K), 1):
                sequences.add((holes1[0], tuple(sorted(holes2)), tuple(sorted(holes3)), holes4[0]))
sequences = list(sequences)
print(len(sequences))

# enforce certain configurations
kept_sequences = set()
for seq in sequences:
    hole1, holes2, holes3, hole4 = seq
    Lhole2, Rhole2 = holes2
    Lhole3, Rhole3 = holes3

    # ignore where second layer holes both require going left or right
    if Rhole2 < hole1 or Lhole2 > hole1:
        continue

    # ignore where third layer holes both require going left or right
    if Rhole3 < Lhole2 or Lhole3 > Rhole2:
        continue

    # ignore where third layer hole could require going left or right for both choices
    if Lhole3 > Lhole2 or Rhole3 < Rhole2:
        continue

    # ignore where any holes match across adjacent sequence positions
    if hole1 in holes2 or hole4 in holes3:
        continue
    if holes2[0] in holes3:
        continue
    if holes2[1] in holes3:
        continue

    kept_sequences.add(seq)
sequences = list(kept_sequences)
print(len(sequences))

# now find sequences where greedy/1-step and 2-step choices differ
kept_sequences = set()
for seq in sequences:
    hole1, holes2, holes3, hole4 = seq
    Lhole2, Rhole2 = holes2
    Lhole3, Rhole3 = holes3
    greedy_dist_L = abs(Lhole2 - hole1)
    greedy_dist_R = abs(Rhole2 - hole1)
    greedy_choice = 0 if greedy_dist_L < greedy_dist_R else 1
    
    one_step_dist_LL = greedy_dist_L + abs(Lhole3 - Lhole2)
    one_step_dist_LR = greedy_dist_L + abs(Rhole3 - Lhole2)
    one_step_dist_L = min(one_step_dist_LL, one_step_dist_LR)
    one_step_dist_RL = greedy_dist_R + abs(Lhole3 - Rhole2)
    one_step_dist_RR = greedy_dist_R + abs(Rhole3 - Rhole2)
    one_step_dist_R = min(one_step_dist_RL, one_step_dist_RR)
    one_step_choice = 0 if one_step_dist_L < one_step_dist_R else 1

    two_step_dist_LL = one_step_dist_LL + abs(hole4 - Lhole3)
    two_step_dist_LR = one_step_dist_LR + abs(hole4 - Rhole3)
    two_step_dist_L = min(two_step_dist_LL, two_step_dist_LR)
    two_step_dist_RL = one_step_dist_RL + abs(hole4 - Lhole3)
    two_step_dist_RR = one_step_dist_RR + abs(hole4 - Rhole3)
    two_step_dist_R = min(two_step_dist_RL, two_step_dist_RR)
    two_step_choice = 0 if two_step_dist_L < two_step_dist_R else 1

    if greedy_choice == one_step_choice and greedy_choice != two_step_choice:
        kept_sequences.add(seq)
sequences = list(kept_sequences)
print(len(sequences))

#%% visualize sequences

import matplotlib.pyplot as plt
def plot_sequence(seq):
    hole1, holes2, holes3, hole4 = seq
    print(seq)
    # y axis is layer index (high y for early layers, low y for later layers)
    # x axis is hole location
    plt.scatter([hole1], [4], color='black')
    plt.scatter(holes2, [3, 3], color='blue')
    plt.scatter(holes3, [2, 2], color='orange')
    plt.scatter([hole4], [1], color='green')
    plt.yticks([1, 2, 3, 4], ['Layer 4', 'Layer 3', 'Layer 2', 'Layer 1'])
    plt.xlabel('Hole location')
    plt.xlim(-1, K)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')

plot_sequence(sequences[100])
