import numpy as np

n = int(input("Enter n"))

A = np.array([list(map(float, input().split())) for i in range(n)])
b = np.array([list(map(float, input().split()))]).reshape(n, 1)

augMat = np.concatenate((A, b), axis=1)

print("Augmented Matrix : ")
print(augMat)

# Forward Phase
for c in range(n):
    subAug = augMat[c:, c:]

    nonZeroRows = np.nonzero(subAug[:, 0])[0]

    if (nonZeroRows.size == 0):
        # There is no nonzero value in this column, This column can't be a pivot column.
        continue
    else:
        firstNonZeroRow = nonZeroRows[0]
        # Scaling
        subAug[0] = subAug[0] / subAug[0, 0]

        # Making Zeros below first non zero row of this column:
        subAug[[0, firstNonZeroRow]] = subAug[[firstNonZeroRow, 0]]
        for r in nonZeroRows[1:]:
            subAug[r] = subAug[r] - (subAug[r, 0] / subAug[0, 0]) * subAug[0]
        augMat[c:, c:] = subAug

print("Echelon Form of the Augmented Matrix: ")
print(augMat)

# Check to see if system is consistent
consistant = True
for r in range(0, n):
    if ((not np.any(augMat[r, :n]))
            and
            (augMat[r, n] != 0)):
        consistant = False

if not consistant:
    print("System is Inconsistant!")
else:
    print("System is Consistant!")
    # Backward Phase
    for c in range(n - 1, -1, -1):
        for r in range(c - 1, -1, -1):
            augMat[r] = augMat[r] - (augMat[r, c] / augMat[c, c]) * augMat[c]
    print(augMat)
