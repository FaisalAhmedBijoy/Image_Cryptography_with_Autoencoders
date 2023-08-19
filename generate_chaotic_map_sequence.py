def logistic_map(x0, l, N):
    """
    Generate a sequence S using the chaotic logistic map with initial parameters x0 and l.
    S = x1, x2, x3, ..., xN, where N is the size of the flattened input image.
    xn+1 = l * xn * (1 - xn)
    """
    S = [x0]
    for i in range(N - 1):
        xn = S[-1]
        xn1 = l * xn * (1 - xn)
        S.append(xn1)
    return S
if __name__=='__main__':
    print('chaotic map generator')
    x0 = 0.5
    l = 3.5
    N = 100 # size of flattened input image
    S = logistic_map(x0, l, N)
    print(S)