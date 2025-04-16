import math
import tabulate
import numpy as np
import matplotlib.pyplot as plt

# Метод Зейделя

def mu(i, j, h, k):
    x = i * h
    y = j * k
    if i > 6 and j < 2:
        return 0
    if i == 0:
        return y
    if i == 8:
        return 2 * (y - 1)
    if j == 4:
        return 2
    return 0

if __name__ == '__main__':
    width = 4
    height = 2
    m = 4
    n = 8
    h = width / n
    k = height / m
    precision = 1e-14
    max_iterations = 1000
    k_sq = k ** 2
    h_star = 1 / (h ** 2)
    k_star = 1 / k_sq
    a_star = -2 * (h_star + k_star)
    grid = [] 
    bad_nodes = [(6, 1), (7, 2)]
    invalid_nodes = [(7, 1)]

    for i in range(m + 1):
        grid.append([0] * (n + 1))
        for j in range(n + 1):
            grid[i][j] = mu(j, i, h, k)

    max_delta = precision + 1
    iterations = 0
    print("Итерации и точность:")
    print("-------------------")
    while iterations < max_iterations and max_delta > precision:
        max_delta = -1
        for j in range(1, m):
            for i in range(1, n):
                if (i, j) in invalid_nodes:
                    continue
                left  = h_star * grid[j][i - 1]
                right = h_star * grid[j][i + 1]
                up    = k_star * grid[j + 1][i]
                down  = k_star * grid[j - 1][i]
                f     = 0
                old   = grid[j][i]
                if (i, j) in bad_nodes:
                    if i == 6:
                        right = 0
                        xw = math.sqrt(1 - (j * k) ** 2) + 4
                        x  = i * h
                        alpha = (xw - x) / h
                        a_star_corrected = -2 * (k_star + h_star / alpha )
                        grid[j][i] = (up + down + 2 * (left / (1 + alpha) + right / (alpha * (1 + alpha)))) / -a_star_corrected
                    else:
                        down = 0
                        yw = math.sqrt(1 - (i * h - 4) ** 2)
                        y  = j * k
                        beta = (y - yw) / k
                        a_star_corrected = -2 * (h_star + k_star / beta)
                        grid[j][i] = (left + right + 2 * (up / (1 + beta) + down / (beta * (1 + beta)))) / -a_star_corrected
                else:
                    grid[j][i] = (f + left + right + up + down) / -a_star
                d = math.fabs(grid[j][i] - old)
                if d > max_delta:
                    max_delta = d
        iterations += 1
        print(f"Итерация {iterations}: максимальное изменение = {max_delta:.3e}")

    print("\nРезультаты:")
    print(f"Всего итераций: {iterations}")
    print(f"Достигнутая точность: {max_delta:.3e}")
    print("-------------------\n")

    x = np.linspace(0, width,  n + 1)
    y = np.linspace(0, height, m + 1)

    X, Y  = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    z    = np.ravel(grid)
    Z    = z.reshape(X.shape)

    cs = ax.contourf(X, Y, Z)     
    cbar = fig.colorbar(cs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim((0, 4))
    ax.set_ylim((0, 2))
    ax.xaxis.set_major_locator(plt.MultipleLocator(h))
    ax.yaxis.set_major_locator(plt.MultipleLocator(k))

    plt.legend()
    plt.grid()

    circle = plt.Circle((4, 0), 1, color='black', clip_on = True)
    ax.add_patch(circle)

    plt.show()