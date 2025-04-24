import math
import tabulate
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Метод Релаксации

def mu(i, j, grid):
    x = i * grid.h
    y = j * grid.m
    if i == 0 or i == grid.n or j == 0 or j == grid.m:
        return math.cos(pi * grid._k * x) * math.cos(pi * grid._k * y)
    return 0

def f(i, j, grid):
    return 0

def u(x, y):
    return 0

class Grid:
    def __init__(self, w, h, n, m, f, mu, _w, k, u = None):
        self.width = w
        self.height = h
        self.w = _w
        self.n = n
        self.m = m
        self.h = w / n
        self.k = h / m
        self.k_sq = self.k ** 2
        self.h_star = 1 / (self.h ** 2)
        self.k_star = 1 / self.k_sq
        self.a_star = 2 * (self.h_star + self.k_star)
        self.grid = []
        self.final_grid = None  # Для хранения последней итерации
        self.f = f
        self.u = u
        self._k = k

        if(self.w > 2 or self.w < 0):
            print('Invalid w.')
            exit(1)

        self.accuracy = []

        self.x = np.linspace(0, self.width,  self.n + 1)
        self.y = np.linspace(0, self.height, self.m + 1)

        self.X, self.Y  = np.meshgrid(self.x, self.y)

        if u:
            self.uv = np.array(u(np.ravel(self.X), np.ravel(self.Y)))
        
        for j in range(m + 1):
            self.grid.append([0] * (n + 1))
            for i in range(n + 1):
                self.set(i, j, mu(i, j, self))

    def __str__(self):
        return self.grid.__str__()

    def get(self, i, j):
        return self.grid[j][i]

    def set(self, i, j, v):
        self.grid[j][i] = v

    def __update(self, i, j):
        prev = self.get(i, j)
        left  = self.h_star * self.get(i - 1, j)
        right = self.h_star * self.get(i + 1, j)
        up    = self.k_star * self.get(i, j + 1)
        down  = self.k_star * self.get(i, j - 1)
        f     = self.f(i, j, self)
        new   = self.w * (left + right + up + down)
        new  += (1 - self.w) * self.a_star * prev + self.w * f
        new  /= self.a_star
        self.set(i, j, new)
        return math.fabs(prev - new)

    def solve(self, precision, max_iterations):
        self.accuracy = []

        max_delta = precision + 1
        iterations = 0

        update_function = self.__update

        while iterations < max_iterations and max_delta > precision:
            max_delta = -1
            for j in range(1, self.m):
                for i in range(1, self.n):
                    d = update_function(i, j)
                    if d > max_delta:
                        max_delta = d
            iterations += 1
            self.accuracy.append(max_delta)

        # Сохраняем финальное состояние сетки
        self.final_grid = [row.copy() for row in self.grid]

        error = None
        if self.u:
            z     = np.ravel(self.grid)
            diffs = np.absolute(np.subtract(z, self.uv))
            error = max(diffs[~np.isnan(diffs)])

        return iterations, max_delta, error

    def plot(self):
        fig  = plt.figure(1)

        ax   = fig.add_subplot(1, 1, 1)

        z    = np.ravel(self.grid)
        Z    = z.reshape(self.X.shape)

        cs = ax.contourf(self.X, self.Y, Z)     
        fig.colorbar(cs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.k))

        plt.legend()
        plt.grid()

        plt.show()
    
    def print_final_iteration_table(self, decimals=4, step=5):
        if self.final_grid is None:
            print("Решение еще не выполнено!")
            return
        
        # Создаем заголовки для таблицы с шагом step
        headers = ["Y\\X"] + [f"{i * self.h:.2f}" for i in range(0, self.n + 1, step)]
        
        # Подготавливаем данные для таблицы с шагом step
        table_data = []
        for j in range(0, self.m + 1, step):
            row = [f"{j * self.k:.2f}"] + [f"{self.final_grid[j][i]:.{decimals}f}" 
                                          for i in range(0, self.n + 1, step)]
            table_data.append(row)
        
        # Выводим таблицу
        print("\nРезультаты последней итерации (с шагом {}):".format(step))
        print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))

if __name__ == '__main__':
    grid = Grid(1, 1, 20, 20, f, mu, 1.5, 1)
    max_iterations = 10000
    iterations, acc, error = grid.solve(1e-14, max_iterations)
    
    # Вывод основной информации
    print(tabulate.tabulate([['Количество итераций', f'{iterations}/{max_iterations}'], 
                             ['Точность', acc], 
                             ['Погрешность', error]], 
                            tablefmt='simple_grid', colalign=('left','right')))
    
    # Вывод таблицы с результатами последней итерации
    grid.print_final_iteration_table(decimals=4, step=1)  
    
    # Построение графика
    grid.plot()
