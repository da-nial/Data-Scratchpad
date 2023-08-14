import numpy as np

n = int(input("Enter n"))
A = np.array([list(map(float, input().split())) for i in range(n)])


# n = 4
# np.random.seed(1)
# A = np.random.randint(-12, 12, 16).reshape(4,4).astype(float)


def LU_Decomposition(A):
    n = len(A)

    # Matrix L:
    L = np.zeros(n * n).reshape(n, n)
    np.fill_diagonal(L, 1)

    # Matrix U:
    U = np.zeros(n * n).reshape(n, n)
    U

    for c in range(n):
        subA = A[c:, c:]

        nonZeroRows = np.nonzero(subA[:, 0])[0]

        if (nonZeroRows.size == 0):
            # There is no nonzero value in this column, This column can't be a pivot column.
            continue
        else:
            firstNonZeroRow = nonZeroRows[0]

            # Making Zeros below first non zero row of this column:
            subA[[0, firstNonZeroRow]] = subA[[firstNonZeroRow, 0]]

            for r in nonZeroRows[1:]:
                k = subA[r, 0] / subA[0, 0]
                subA[r] = subA[r] - (k) * subA[0]
                U[c:, c:] = subA
                L[c + r, c] = k
    print("U: \n" + str(U))
    print("L: \n" + str(L))
    print("\n\n\n")
    return (L, U)


def forward_substitution(ins):
    L = ins[0]

    n = len(L)

    b = ins[1].reshape(n, 1)

    augMat = np.concatenate((L, b), axis=1)

    for c in range(n):
        subAug = augMat[c:, c:]

        nonZeroRows = np.nonzero(subAug[:, 0])[0]

        if (nonZeroRows.size == 0):
            # There is no nonzero value in this column, This column can't be a pivot column.
            continue
        else:
            firstNonZeroRow = nonZeroRows[0]

            # Making Zeros below first non zero row of this column:
            #BECAUSE OF THE FACT THAT THE OTHER ELEMENTS ARE ZERO, WE COULD ONLY SUBSTRACT PIVOT AND LAST COLUMN
            subAug[[0, firstNonZeroRow]] = subAug[[firstNonZeroRow, 0]]
            for r in nonZeroRows[1:]:
                subAug[r] -= (subAug[r, 0] / subAug[0, 0]) * subAug[0]
            augMat[c:, c:] = subAug
    y = augMat[:, n]

    #     print('L: ' + str(L))
    #     print('b: ' + str(b))
    #     print('y: ' + str(y))
    return (y)


def backward_substitution(ins):
    U = ins[0]
    n = len(U)
    y = ins[1].reshape(n, 1)

    augMat = np.concatenate((U, y), axis=1)

    for c in range(n - 1, -1, -1):
        # BECAUSE OF THE FACT THAT THE OTHER ELEMENTS ARE ZERO, WE COULD ONLY SUBSTRACT PIVOT AND LAST COLUMN
        augMat[c] /= augMat[c, c]
        for r in range(c - 1, -1, -1):
            augMat[r] = augMat[r] - (augMat[r, c] / augMat[c, c]) * augMat[c]
    x = augMat[:, n]

    #     print("U: " + str(U))
    #     print("y: " + str(y))
    #     print("x: " + str(x))
    #     print("\n\n")
    return x


A_inv = np.zeros(n * n).reshape(n, n)

I = np.identity(n)

L, U = LU_Decomposition(A)
for i in range(len(I)):
    ins = [L, I[:, i]]
    y = forward_substitution(ins)
    ins = [U, y]
    x = backward_substitution(ins)
    A_inv[:, i] = x

print("A_Inverse: \n" + str(A_inv))