import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Функция для определения системы уравнений
def system(state, t, a, b, c, d):
    x, y = state
    dxdt = a * x + b * y
    dydt = c * x + d * y
    return [dxdt, dydt]

# Функция для построения фазового портрета
def plot_phase_portrait(a, b, c, d, x_range=(-5, 5), y_range=(-5, 5), grid_size=20, t_max=10):
    # Создаем сетку для векторного поля
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # Вычисляем компоненты векторного поля
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            dxdt, dydt = system([X[i, j], Y[i, j]], 0, a, b, c, d)
            U[i, j] = dxdt
            V[i, j] = dydt

    # Нормализуем векторы для одинакового размера
    norm = np.sqrt(U**2 + V**2)
    U_norm = U / norm
    V_norm = V / norm

    # Строим фазовый портрет
    plt.figure(figsize=(8, 6))

    # Рисуем векторное поле
    plt.quiver(X, Y, U_norm, V_norm, color='gray', alpha=0.5, label='Vector field')
    # Рисуем траектории
    t = np.linspace(0, t_max, 500)
    initial_conditions = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [2, 1],
        [-2, -1],
        [1, 1],
        [-1, -1]
    ]
    for ic in initial_conditions:
        states = odeint(system, ic, t, args=(a, b, c, d))
        plt.plot(states[:, 0], states[:, 1], label=f'x0={ic[0]}, y0={ic[1]}')

    # Настраиваем график
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Фазовый портрет системы')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

# Функция для определения типа точки равновесия
def equilibrium_type(a, b, c, d, tol=1e-10):
    # Матрица Якоби
    J = np.array([[a, b], [c, d]])
    # Собственные значения
    eigenvalues = np.linalg.eigvals(J)
    print("Собственные значения (до обработки):", eigenvalues)

    # Заменяем очень маленькие вещественные части на ноль
    eigenvalues = np.where(np.abs(np.real(eigenvalues)) < tol, 1j * np.imag(eigenvalues), eigenvalues)
    print("Собственные значения (после обработки):", eigenvalues)

    # Определяем тип точки равновесия
    if np.all(np.real(eigenvalues) < 0) and np.all(np.imag(eigenvalues) != 0):
        return "Устойчивый фокус"
    elif np.all(np.real(eigenvalues) > 0) and np.all(np.imag(eigenvalues) != 0):
        return "Неустойчивый фокус"
    elif np.all(np.real(eigenvalues) < 0):
        return "Устойчивый узел"
    elif np.all(np.real(eigenvalues) > 0):
        return "Неустойчивый узел"
    elif np.any(np.real(eigenvalues) < 0) and np.any(np.real(eigenvalues) > 0):
        return "Седло"
    elif np.all(np.abs(np.real(eigenvalues)) < tol) and np.all(np.imag(eigenvalues) != 0):
        return "Центр (Эллипс)"
    else:
        return "Неопределенный тип"

# a, b, c, d = 3, -10, 1, -3
a, b, c, d = map(int, input("Введите a, b, c, d: ").split())
print(f"Система имеет вид: \nx` = {a}x + {b}y\ny` = {c}x + {d}y")
print("Тип точки равновесия:", equilibrium_type(a, b, c, d))
plot_phase_portrait(a, b, c, d)