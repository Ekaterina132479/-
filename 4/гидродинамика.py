from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def create_grid(length, time_slice, n, m, u0, mu):
    grid = []

    h = length / n
    t = time_slice / m

    for j in range(m):
        grid.append([0] * n)

    for i in range(n):
        grid[0][i] = u0(i * h)

    for j in range(m):
        grid[j][0] = mu(j * t)

    return grid, h, t

# функции для начальных условий
def u0_0(x):
    return float(x >= 1 and x <= 2) # прямоугольный импульс


def u0_1(x):
    if x < 1 or x > 2:
        return 0
    elif x >= 1 and x < 1.5:
        return 2 * (x - 1)
    else:
        return 1 - 2 * (x - 1.5) # треугольный импульс

def u0_2(x):
    if x < 1 or x > 2:
        return 0
    else:
        return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2)) # синусоидальный импульс
    
def u0_3(x):
    if x < 0 or x > 1:
        return 0
    else:
        return 1 - x**2  # функция для половины параболы
    
# функции для граничных условий
def mu(t):
    return u0_3(-t/a)

# Вариант 1: u_0(-a*t)

# Вариант 2: u_0(-t/a)

# Вариант 3: 0 

def solve1(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t])):
            grid[t][x] = grid[t - 1][x] - a * tau * ((grid[t - 1][x] - grid[t - 1][x - 1]) / h)
    return grid

def solve2(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t]) - 1):
            grid[t][x] = grid[t - 1][x] - a * tau * ((grid[t - 1][x + 1] - grid[t - 1][x - 1]) / (2 * h))
    return grid

def generate_precise(u0, a, i, h, n, tau):
    grid = []
    for x in range(n):
        grid.append(u0(x * h - i * tau * a))
    return grid

a = 2
w = 15
h = 10
n = w * 10
m = h * 20

# tau / h < 0.5

if __name__ == '__main__':
    target = u0_3  # тут задаётся начальное условие

    grid1, hx, tau = create_grid(w, h, n, m, target, mu)
    grid2, hx, tau = create_grid(w, h, n, m, target, mu)

    c = abs(a) * tau / hx
    if c > 1:
        print('Doesnt converge: c = ', c)
        print(tau, hx)
        exit(1)

    grid1 = solve1(grid1, a, hx, tau)
    grid2 = solve2(grid2, a, hx, tau)

    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        ax.plot(x, grid1[frame], '-', label = 'method 1')
        #ax.plot(x, grid2[frame], '-', label = 'method 2')
        ax.plot(x, generate_precise(target, a, frame, hx, n, tau), '-', label = 'precise')
        plt.legend()

    ani = animation.FuncAnimation(fig, update, frames=m, interval=30)

    plt.show()