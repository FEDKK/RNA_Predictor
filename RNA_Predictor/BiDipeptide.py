import numpy as np

def BiDipeptide(selected_sequence):
    print('Reading RNA file...')

    sequence = str(selected_sequence)

    print('Start extracting RNA features...')

    AA = 'AUGC'
    M = len(sequence)
    F1 = np.zeros((4 * 4, M - 1))  # Record the number of occurrences of each two amino acids at each position
    F2 = np.zeros((4 * 4, M - 1))

    for k in range(M - 1):
        s = sequence[k]
        t = sequence[k + 1]
        i = AA.index(s)
        j = AA.index(t)
        F1[j + (i * 4), k] += 1

    F1 /= len(sequence)

    PPT = np.zeros(2 * (M - 1))

    for k in range(M - 1):
        s = sequence[k]
        t = sequence[k + 1]
        i = AA.index(s)
        j = AA.index(t)
        PPT[k] = F1[j + (i * 4), k]

    print('RNA features extraction complete...')

    return PPT


