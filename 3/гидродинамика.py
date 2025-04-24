import math
import tabulate
import numpy as np
import matplotlib.pyplot as plt

# Метод Сопряженных Градиентов

def mu(i, j, grid):
    x = i * grid.h
    y = j * grid.k

    if i == 0 and j >= grid.m // 4:
        return y ** 2

    if i == grid.n:
        return y ** 2 + 4

    if j == grid.m // 4 and i <= grid.n // 2:
        return x ** 2 + 0.0625

    if j == grid.m:
        return x ** 2 + 1

    if i == grid.n // 2 and j < grid.m // 4:
        return y ** 2 + 1

    if j == 0 and i >= grid.n // 2:
        return x ** 2

    if j < grid.m // 4 and i < grid.n // 2:
        return np.nan

    return 0

def f(i, j, grid):
    return -4

def u(x, y):
    return x**2 + y**2

class Grid:
    def __init__(self, w, h, n, m, f, mu, u = None):
        self.width = w
        self.height = h
        self.n = n
        self.m = m
        self.h = w / n
        self.k = h / m
        self.k_sq = self.k ** 2
        self.h_star = 1 / (self.h ** 2)
        self.k_star = 1 / self.k_sq
        self.a_star = -2 * (self.h_star + self.k_star)
        self.grid = []
        self.r = []
        self.z = []
        self.H = []
        self.c = 0
        self.d = 0
        self.alpha = 0
        self.beta  = 0
        self.f = f
        self.u = u
        self.final_grid = None  # Добавлено для хранения последней итерации

        self.accuracy = []

        self.x = np.linspace(0, self.width,  self.n + 1)
        self.y = np.linspace(0, self.height, self.m + 1)

        self.X, self.Y  = np.meshgrid(self.x, self.y)

        if u:
            self.uv = np.array(u(np.ravel(self.X), np.ravel(self.Y)))
        
        for j in range(m + 1):
            self.grid.append([0] * (n + 1))
            self.r.append([0] * (n + 1))
            self.z.append([0] * (n + 1))
            self.H.append([0] * (n + 1))
            for i in range(n + 1):
                self.set(i, j, mu(i, j, self))
                self.r[j][i] = 0
                self.z[j][i] = 0
                self.H[j][i] = 0
                
    def __str__(self):
        return self.grid.__str__()

    def get(self, i, j):
        return self.grid[j][i]

    def set(self, i, j, v):
        self.grid[j][i] = v

    def __update(self, i, j):
        prev = self.get(i, j)
        new   = prev + self.alpha * self.H[j][i]
        self.set(i, j, new)
        return math.fabs(prev - new)

    def __grid_cycle(self, func):
        for j in range(1, self.m // 4 + 1):
            for i in range(self.n // 2 + 1, self.n):
                func(i, j)
        for j in range(self.m // 4 + 1, self.m):
            for i in range(1, self.n):
                func(i, j)

    def __calc_r(self, i, j):
        prev  = self.a_star * self.get(i, j)
        left  = self.h_star * self.get(i - 1, j)
        right = self.h_star * self.get(i + 1, j)
        up    = self.k_star * self.get(i, j + 1)
        down  = self.k_star * self.get(i, j - 1)
        f     = self.f(i, j, self)
        self.r[j][i] = f + prev + left + right + up + down

    def __calc_z(self, i, j):
        prev  = self.a_star * self.H[j][i]
        left  = self.h_star * self.H[j][i - 1]
        right = self.h_star * self.H[j][i + 1]
        up    = self.k_star * self.H[j + 1][i]  
        down  = self.k_star * self.H[j - 1][i]
        self.z[j][i] = prev + left + right + up + down

    def __calc_alpha(self, i, j):
        self.c += self.r[j][i] * self.r[j][i]
        self.d += self.H[j][i] * self.z[j][i]

    def __calc_beta(self, i, j):
        self._c += self.r[j][i] * self.r[j][i]

    def __calc_h(self, i, j):
        self.H[j][i] = -self.r[j][i] + self.beta * self.H[j][i]

    def __reset_counters(self):
        self.c  = 0
        self.d  = 0
        self._c = 0

    def __reset(self):
        self.r = []
        self.z = []
        self.H = []
        self.__reset_counters()
        for j in range(self.m + 1):
            self.r.append([0] * (self.n + 1))
            self.z.append([0] * (self.n + 1))
            self.H.append([0] * (self.n + 1))
            for i in range(self.n + 1):
                self.set(i, j, mu(i, j, self))
                self.r[j][i] = 0
                self.z[j][i] = 0
                self.H[j][i] = 0
        self.__grid_cycle(self.__calc_r)
        self.__grid_cycle(self.__calc_h)
        self.__grid_cycle(self.__calc_z)

    

    def solve(self, precision, max_iterations):
        self.accuracy = []

        max_delta = precision + 1
        iterations = 0

        update_function = self.__update

        self.__grid_cycle(self.__calc_r)
        self.__grid_cycle(self.__calc_h)
        self.__grid_cycle(self.__calc_z)

        while iterations < max_iterations and max_delta > precision:
            self.__reset_counters()
            self.__grid_cycle(self.__calc_alpha)
            self.alpha = self.c / self.d
            max_delta = -1
            for j in range(1, self.m // 4 + 1):
                for i in range(self.n // 2 + 1, self.n):
                    d = update_function(i, j)
                    if d > max_delta:
                        max_delta = d
            for j in range(self.m // 4 + 1, self.m):
                for i in range(1, self.n):
                    d = update_function(i, j)
                    if d > max_delta:
                        max_delta = d
            iterations += 1
            self.accuracy.append(max_delta)

            self.__grid_cycle(self.__calc_r)
            self.__grid_cycle(self.__calc_beta)
            self.beta = self._c / self.c
            if self.beta < 1e-6:
                self.__reset()
            else:
                self.__grid_cycle(self.__calc_h)
                self.__grid_cycle(self.__calc_z)
        
        # Сохраняем финальное состояние сетки
        self.final_grid = [row.copy() for row in self.grid]

        error = None
        if self.u:
            z = np.ravel(self.grid)
            diffs = np.absolute(np.subtract(z, self.uv))
            error = max(diffs[~np.isnan(diffs)])

        return iterations, max_delta, error

    def print_final_grid_table(self, step_x=5, step_y=5, decimals=4):
        """Выводит таблицу с результатами последней итерации"""
        if self.final_grid is None:
            print("Решение еще не выполнено!")
            return
        
        # Создаем заголовки для таблицы
        headers = ["Y\\X"] + [f"{i*self.h:.2f}" for i in range(0, self.n+1, step_x)]
        
        # Подготавливаем данные для таблицы
        table_data = []
        for j in range(0, self.m+1, step_y):
            row = [f"{j*self.k:.2f}"] + [
                f"{self.final_grid[j][i]:.{decimals}f}" if not np.isnan(self.final_grid[j][i]) else "NaN" 
                for i in range(0, self.n+1, step_x)
            ]
            table_data.append(row)
        
        # Выводим таблицу
        print("\nРезультаты последней итерации:")
        print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))
        
    def plot(self):
        fig  = plt.figure()
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

        rect = plt.Rectangle((0, 0), 1, 0.25, linewidth=1, edgecolor='none', facecolor='black')
        ax.add_patch(rect)

        plt.show()
    
if __name__ == '__main__':
    grid = Grid(2, 1, 20, 20, f, mu, u)
    max_iterations = 10000
    iterations, acc, error = grid.solve(1e-14, max_iterations)
    
    # Вывод основной информации
    print(tabulate.tabulate([
        ['Количество итераций', f'{iterations}/{max_iterations}'],
        ['Точность', acc],
        ['Погрешность', error]
    ], tablefmt='simple_grid', colalign=('left','right')))
    
    # Вывод таблицы с результатами последней итерации
    grid.print_final_grid_table(step_x=1, step_y=1,decimals=3)  
    
    # Построение графика
    grid.plot()