
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from math import *

# область и сетка
n = 50  # количество узлов сетки
n_step = 0
x_left = 0  # левая граница
x_right = 1  # правая граница
n_comp = 3  # Количество компонент газовой смеси
# a = 0.0  # Левый конец отрезка по r
# b = 1.0  # Правый конец отрезка по r

# параметры левого и правого газа
U_left = 0.0
P_left = 1000
Rho_left = 1.0

U_center = 0.0
P_center = 0.01
Rho_center = 1.0

U_right = 0.0
P_right = 100
Rho_right = 1.0

# координата границы между газами
x0 = 0.1
x0_P = 0.1  # Координаты точки разрыва давления
x0_U = 0.1  # Координаты точки разрыва скорости
x0_Rho = 0.1  # Координаты точки разрыва плотности

x1 = 0.9
x1_P = 0.9  # Координаты точки разрыва давления
x1_U = 0.9  # Координаты точки разрыва скорости
x1_Rho = 0.9  # Координаты точки разрыва плотности

gamma_left = 1.4  # показатель адиабаты слева
gamma_right = 1.4  # показатель адиабаты справа
gamma_center = 1.4
init_type = 0  # Тип согласования НУ; 0 - консервативные = полусуммам потоковых;
               # 1 - консервативные = потоковые слева; 2 - консервативные = потоковые справа

# Число Куранта
Cour = 0.0001  # Начальное число Куранта
Cour_incr_freq = 1000  # Раз в сколько шагов увеличивать Куранта

# Время
FullTime = 0.038  # Время расчета
time = 0          # Текущее время

R_gas = 8.314463  # Универсальная газовая постоянная
eps_u = 1e-10  # малое число для сравнения с 0

h = (x_right - x_left) / n  # шаг
x = np.arange(n + 1) * h  # координаты узлов

# gamma
gamma = np.zeros(n_comp)  # показатель адиабаты отдельных компонент смеси
c_temp = np.zeros(n_comp)  # Cv отдельных компонент смеси
gamma_tmp = np.zeros(n_comp)

gamma_tmp[0] = gamma_left
gamma_tmp[1] = gamma_center
gamma_tmp[2] = gamma_right

gamma = gamma_tmp  # показатель адиабаты
c_temp = R_gas / (gamma - 1)

# НУ консервативные
# U_cons = np.zeros(n)
# Rho_cons = np.zeros(n)  # консервативные плотности на целых слоях
# Rho_comp_cons = np.zeros((n, n_comp))  # консервативные плотности компонент на целых слоях
# P_cons = np.zeros(n)
# E_cons = np.zeros(n)
R_speed_cell = np.zeros(n)  # скорость переноса инвариантов R в центрах ячеек
Q_speed_cell = np.zeros(n)  # скорость переноса инвариантов Q в центрах ячеек
S_speed_cell = np.zeros(n)  # скорость переноса инвариантов S в центрах ячеек
G_cell = np.zeros(n)        # Коэффициенты для локальных инвариантов Римана
C_cell = np.zeros(n)        # Скорости звука в центрах пространственно-временных ячейках
R_cons_half = np.zeros(n)   # Консервативные инварианты R на полуцелых слоях
Q_cons_half = np.zeros(n)   # Консервативные инварианты Q на полуцелых слоях
S_cons_half = np.zeros(n)   # Консервативные инварианты S на полуцелых слоях

U_cons_half = np.zeros(n)  # Консервативные скорости на полуцелых слоях
Ksi_cons_half = np.zeros((n, n_comp))  # Консервативные инварианты Ksi на полуцелых слоях
Rho_cons_half = np.zeros(n)  # Консервативные плотности на полуцелых слоях
Rho_comp_cons_half = np.zeros((n, n_comp))  # Консервативные плотности компонент на полуцелых слоях
P_cons_half = np.zeros(n)  # Консервативные давления на полуцелых слоях
E_cons_half = np.zeros(n)  # Консервативные полные энергии на полуцелых слоях
gamma_loc_cons = np.zeros(n)  # Консервативные эффективные показатели адиабаты на целых слоях
gamma_loc_cons_half = np.zeros(n)  # Консервативные эффективные показатели адиабаты на полуцелых слоях
max_c = 0.0  # Максимальное по модулю собственное число

# НУ потоковые переменные и инварианты
U_flux = np.zeros(n + 1)
Rho_flux = np.ones(n + 1)  # Потоковые плотности
Rho_comp_flux = np.zeros((n + 1, n_comp))  # Потоковые плотности компонент
Ksi_comp_flux = np.zeros((n + 1, n_comp))  # Потоковые концентрации компонент
P_flux = np.ones(n + 1) * 0.01
# E_flux = np.zeros(n + 1)
R_flux = np.zeros(n + 1)
Q_flux = np.zeros(n + 1)
S_flux = np.zeros(n + 1)
# gamma_loc_flux = np.zeros(n + 1)  # Потоковые эффективные показатели адиабаты

# Диссипаторы
dissip_R = 0.0
dissip_Q = 0.0
dissip_S = 0.0
dissip_Ksi = 0.0

# флаг включения монотонизации
monotize = True
RQ_right_monot = 0.1  # Учет правой части в монотонизации инвариантов R и Q (0 - не учитывается, 1 - полностью учитывается)
S_right_monot = 0.5
Ksi_right_monot = 0.0
# Rho_tmp_vec = np.zeros(n)

# граничные условия
bc_left_v = True  # Граничное условие u = U_left на левой границе
bc_left_p = False  # Граничное условие p = P_left на левой границе
bc_right_v = True  # Граничное условие u = U_right на правой границе
bc_right_p = False  # Граничное условие p = P_right на правой границе
bc_left_R = False  # Неотражающее граничное условие на левой границе
bc_right_Q = False  # Неотражающее граничное условие на правой границе

# задание начальных потоковых
mask1 = np.bitwise_and(x > x0_Rho, x <= x1_Rho)

Rho_comp_flux[x <= x0_Rho, 0] = Rho_left
Rho_comp_flux[x <= x0_Rho, 1] = 0.0
Rho_comp_flux[x <= x0_Rho, 2] = 0.0

Rho_comp_flux[mask1, 0] = 0.0
Rho_comp_flux[mask1, 1] = Rho_center
Rho_comp_flux[mask1, 2] = 0.0

Rho_comp_flux[x > x1_Rho, 0] = 0.0
Rho_comp_flux[x > x0_Rho, 1] = 0.0
Rho_comp_flux[x > x0_Rho, 2] = Rho_right

# mask1 = np.bitwise_and(x0_Rho < x, x <= 1)
# Rho_flux[mask1] = Rho_right
P_flux[:n + 1] = (x <= x0_P) * P_left + mask1 * P_center + (x > x1_P) * P_right
U_flux[:n + 1] = (x <= x0_U) * U_left + mask1 * U_center + (x > x1_U) * U_right

up = np.sum(gamma * c_temp * Rho_comp_flux, axis=1)
down = np.sum(c_temp * Rho_comp_flux, axis=1)
gamma_loc_flux = up / down
E_flux = P_flux / (Rho_flux * (gamma_loc_flux - 1)) + 0.5 * U_flux * U_flux

P_left = P_flux[0]
P_right = P_flux[n]
U_left = U_flux[0]
U_right = U_flux[n]
Rho_left = Rho_flux[0]
Rho_right = Rho_flux[n]

# Задание начальных консервативных

if init_type == 0:
    P_cons = 0.5 * (P_flux[:n] + np.roll(P_flux, -1)[:n])
    U_cons = 0.5 * (U_flux[:n] + np.roll(U_flux, -1)[:n])
    Rho_cons = 0.5 * (Rho_flux[:n] + np.roll(Rho_flux, -1)[:n])
    Rho_comp_cons = 0.5 * (Rho_comp_flux[:n, :] + np.roll(Rho_comp_flux, -1)[:n, :])

elif init_type == 1:
    P_cons = P_flux[:n]
    U_cons = U_flux[:n]
    Rho_cons = Rho_flux[:n]

    Rho_comp_cons = np.copy(Rho_comp_flux[:n])

else:
    P_cons = np.roll(P_flux, -1)[:n]
    U_cons = np.roll(P_flux, -1)[:n]
    Rho_cons = np.roll(P_flux, -1)[:n]
    Rho_comp_cons = np.roll(Rho_comp_flux, -1)[:n]

up = np.sum(gamma * c_temp * Rho_comp_cons, axis=1)
down = np.sum(c_temp * Rho_comp_cons, axis=1)
Rho_comp_cons[:n] = 0.5 * (Rho_comp_flux[:n] + np.roll(Rho_comp_flux, -1, axis=0)[:n])
gamma_loc_cons = up / down

E_cons = P_cons / (Rho_cons * (gamma_loc_cons - 1)) + 0.5 * U_cons * U_cons

C_cell = np.sqrt(gamma_loc_cons * P_cons / Rho_cons)

C_left = np.sqrt(P_left * gamma_loc_cons[0] / Rho_left)
C_right = np.sqrt(P_right * gamma_loc_cons[n - 1] / Rho_right)
S_left_bc = P_left - P_left * gamma_loc_cons[0]
S_right_bc = P_right - P_right * gamma_loc_cons[n - 1]
G_left = 1 / np.sqrt(P_left * gamma_loc_cons[0] * Rho_left)
G_right = 1 / np.sqrt(P_right * gamma_loc_cons[n - 1] * Rho_right)
R_left_bc = U_left + P_left * G_left
Q_right_bc = U_right - P_right * G_right

# шаг по времени

min_tau_0 = np.inf

min_tau_1 = np.min(Cour * h / (np.sqrt(gamma_loc_cons * P_cons / Rho_cons) + np.abs(U_cons)))

flux = (np.roll(Rho_comp_flux, -1, axis=0)[:n] * np.roll(U_flux.reshape(-1, 1), -1)[:n] -
        Rho_comp_flux[:n] * U_flux.reshape(-1, 1)[:n]) / (np.roll(x, -1)[:n] - x[:n]).reshape(-1, 1)

idx = np.bitwise_and(flux > 0, Rho_comp_cons > 0)

tau_tmp = Rho_comp_cons[idx] / flux[idx]
if tau_tmp:
    min_tau_2 = np.min(tau_tmp)
else:
    min_tau_2 = np.inf

min_tau = min(min_tau_0, min_tau_1, min_tau_2)
tau = min_tau

t_start = tm.perf_counter()

t_steps = []
while time < FullTime:
    t_step_start = tm.perf_counter()

    if FullTime < time + tau:
        tau = FullTime - time
    # tau = min_tau

    time += tau
    # if time > 0.029:
    #     print('Вы здесь')

    # Нахождение промежуточных значений для каждой ячейки

    Rho_comp_cons_half = Rho_comp_cons - 0.5 * tau * (np.roll(U_flux.reshape(-1, 1), -1)[:n] * \
                                                      np.roll(Rho_comp_flux, -1, axis=0)[:n]
                                                      - U_flux.reshape(-1, 1)[:n] * Rho_comp_flux[:n]) / \
                         (np.roll(x, -1)[:n] - x[:n]).reshape(-1, 1)
    Rho_cons_half = np.sum(Rho_comp_cons_half, axis=1)

    U_cons_half = Rho_cons * U_cons - 0.5 * tau * (np.roll(Rho_flux, -1)[:n] * np.roll(U_flux, -1)[:n] ** 2 +
                                                   np.roll(P_flux, -1)[:n] -
                                                   Rho_flux[:n] * U_flux[:n] ** 2 - P_flux[:n]) \
                  / (np.roll(x, -1)[:n] - x[:n])
    U_cons_half /= Rho_cons_half

    E_cons_half = Rho_cons * E_cons - 0.5 * tau * (np.roll(Rho_flux, -1)[:n] *
                                                   np.roll(U_flux, -1)[:n] * np.roll(E_flux, -1)[:n]
                                                   + np.roll(P_flux, -1)[:n] *
                                                   np.roll(U_flux, -1)[:n] - Rho_flux[:n] * U_flux[:n] *
                                                   E_flux[:n] - P_flux[:n] * U_flux[:n]) / \
                  (np.roll(x, -1)[:n] - x[:n])

    E_cons_half /= Rho_cons_half

    up = np.sum(gamma * c_temp * Rho_comp_cons_half, axis=1)
    down = np.sum(c_temp * Rho_comp_cons_half, axis=1)
    gamma_loc_cons_half = up / down
    P_cons_half = (E_cons_half - 0.5 * U_cons_half * U_cons_half) * Rho_cons_half * (gamma_loc_cons_half - 1)

    # Вторая фаза алгоритма

    P_flux_new = np.zeros(n + 1)
    U_flux_new = np.zeros(n + 1)
    Rho_flux_new = np.zeros(n + 1)
    Ksi_flux_new = np.zeros((n + 1, n_comp))
    gamma_loc_flux_new = np.zeros(n + 1)
    # пересчет собственных значений и консервативных инвариантов

    C_cell = np.sqrt(gamma_loc_cons_half * P_cons_half / Rho_cons_half)

    R_speed_cell = U_cons_half + C_cell    # lambda3
    Q_speed_cell = U_cons_half - C_cell    # lambda1
    S_speed_cell = U_cons_half             # lambda2
    G_cell = 2 * np.sqrt(P_cons_half / (Rho_cons_half * gamma_loc_cons))

    R_cons_half = U_cons_half + G_cell                                 # w3 = u+g
    Q_cons_half = U_cons_half - G_cell                                 # w1 = u-g
    S_cons_half = np.log(P_cons_half / Rho_cons_half**gamma_loc_cons)  # w2 = ln p/rho^gamma
    Ksi_cons_half = Rho_comp_cons_half / Rho_cons_half.reshape(-1, 1)

    # -------------------------------------------------------------------------------------------------------------------
    # цикл по всем внутренним граням, расчет новых потоковых значений
    # расчет инварианта R, который приходит справа
    # -------------------------------------------------------------------------------------------------------------------

    speed = np.zeros(n + 1)
    speed[1:n] = np.roll(R_speed_cell, 1)[1:n] * R_speed_cell[1:n]
    mask_speed_R = np.zeros(n + 1)
    mask_speed_R[1:n] = np.bitwise_and(speed[1:n] > 0, R_speed_cell[1:n] < 0)
    mask_speed_R = np.array(mask_speed_R, dtype=bool)

    min1 = np.zeros(n + 1)
    max1 = np.zeros(n + 1)

    R_left = U_flux[mask_speed_R] + G_cell[mask_speed_R[:n]]
    R_right = np.roll(U_flux, -1)[mask_speed_R] + G_cell[mask_speed_R[:n]]
    R_cent = U_cons[mask_speed_R[:n]] + G_cell[mask_speed_R[:n]]
    R_flux1 = np.zeros(n + 1)
    R_flux1[mask_speed_R] = (2 * R_cons_half[mask_speed_R[:n]] - (1 - dissip_R) * R_right) / (1 + dissip_R)

    if np.any(mask_speed_R):

        if monotize:
            Q_rh = RQ_right_monot * (
                    (R_cons_half[mask_speed_R[:n]] - R_cent) / (tau * 0.5) + R_speed_cell[mask_speed_R[:n]] * (
                    R_right - R_left) / (np.roll(x, -1)[mask_speed_R] - x[mask_speed_R]))
            min2 = np.min(np.vstack([R_left, R_right, R_cons_half[mask_speed_R[:n]]]), axis=0) + tau * Q_rh
            max2 = np.max(np.vstack([R_left, R_right, R_cons_half[mask_speed_R[:n]]]), axis=0) + tau * Q_rh
            min1[mask_speed_R] = min2
            max1[mask_speed_R] = max2

        G_R1 = G_cell[mask_speed_R[:n]]

    else:
        G_R1 = 0

    # R приходит слева
    mask_speed_R2 = np.zeros(n + 1)
    mask_speed_R2[1:n] = np.bitwise_and(speed[1:n] > 0, R_speed_cell[1:n] > 0)
    mask_speed_R2 = np.array(mask_speed_R2, dtype=bool)

    R_left2 = np.array(
        np.roll(U_flux, 1)[mask_speed_R2] + np.roll(G_cell, 1)[mask_speed_R2[:n]])

    R_right2 = U_flux[mask_speed_R2] + np.roll(G_cell, 1)[mask_speed_R2[:n]]

    R_cent2 = np.array(
        np.roll(U_cons, 1)[mask_speed_R2[:n]] + np.roll(G_cell, 1)[mask_speed_R2[:n]])

    R_flux2 = np.zeros(n + 1)
    R_flux2[mask_speed_R2] = (2 * np.roll(R_cons_half, 1)[mask_speed_R2[:n]] -
                              (1 - dissip_R) * R_left2) / (1 + dissip_R)

    if np.any(mask_speed_R2):

        if monotize:
            Q_rh = RQ_right_monot * (
                    (np.roll(R_cons_half, 1)[mask_speed_R2[:n]] - R_cent2) / (tau * 0.5) +
                    np.roll(R_speed_cell, 1)[mask_speed_R2[:n]] *
                    (R_right2 - R_left2) / (x[mask_speed_R2] - np.roll(x, 1)[mask_speed_R2]))

            min2 = np.min(np.vstack([R_left2, R_right2, np.roll(R_cons_half, 1)[mask_speed_R2[:n]]]),
                          axis=0) + tau * Q_rh
            max2 = np.max(np.vstack([R_left2, R_right2, np.roll(R_cons_half, 1)[mask_speed_R2[:n]]]),
                          axis=0) + tau * Q_rh
            min1[mask_speed_R2] = min2
            max1[mask_speed_R2] = max2

        G_R2 = np.roll(G_cell, 1)[mask_speed_R2[:n]]

    else:
        G_R2 = 0

    # для сходящихся характеристик
    mask_speed_R3 = np.bitwise_and(R_speed_cell[1:n] < 0, np.roll(R_speed_cell, 1)[1:n] > 0)
    R_left3 = np.roll(U_cons, 1)[mask_speed_R3] + np.roll(G_cell[mask_speed_R3[:n]]) # узел i - 1 на слое n
    R_right3 = U_cons[mask_speed_R3[:n]] + G_cell[mask_speed_R3[:n]]
    R_center3 = U_flux[mask_speed_R3] + G_cell[mask_speed_R3[:n]]  # узел i на n слое

    R_flux3 = np.zeros(n+1)
    R_flux3[mask_speed_R3] = np.roll(R_cons_half, 1)[mask_speed_R3[:n]] + R_cons_half[mask_speed_R3[:n]] - R_center3

    # TODO: Подумать над monotize

    if np.any(mask_speed_R3):

        if monotize:
            Q_rh = RQ_right_monot * (
                    (np.roll(R_cons_half, 1)[mask_speed_R3[:n]] - R_right3) / (tau * 0.5) +
                    np.roll(R_speed_cell, 1)[mask_speed_R3[:n]] *
                    (R_right3 - R_left3) / (x[mask_speed_R3] - np.roll(x, 1)[mask_speed_R3]))

            min3 = np.min(np.vstack([R_left3, R_right3, np.roll(R_cons_half, 1)[mask_speed_R3[:n]]]),
                          axis=0) + tau * Q_rh
            max3 = np.max(np.vstack([R_left3, R_right3, np.roll(R_cons_half, 1)[mask_speed_R3[:n]]]),
                          axis=0) + tau * Q_rh
            min1[mask_speed_R3] = min3
            max1[mask_speed_R3] = max3

        G_R2 = np.roll(G_cell, 1)[mask_speed_R2[:n]]

    else:
        G_R2 = 0


    # для расходящихся характеристик




    # монотонизация решения
    R_flux12 = R_flux1 + R_flux2

    if monotize:
        R_flux_monot = np.bitwise_and(R_flux12[1:n] <= max1[1:n], R_flux12[1:n] >= min1[1:n]) * R_flux12[1:n] + (
                R_flux12[1:n] > max1[1:n]) * max1[1:n] + (R_flux12[1:n] < min1[1:n]) * min1[1:n]
        R_flux12[1:n] = R_flux_monot

    mask_speed_R3 = np.zeros(n + 1)
    mask_speed_R3[1:n] = (speed[1:n] < 0)
    mask_speed_R3 = np.array(mask_speed_R3, dtype=bool)

    # характеристики расходятся

    # mask_speed_R3 = np.zeros(n + 1)
    # mask_speed_R3[1:n] = np.bitwise_and(R_speed_cell[1:n] >= 0, np.roll(R_speed_cell, 1)[1:n] <= 0)
    # полусумма плотностей и давлений U (i,n )+ 1/rho*c(i+-1/2) *P(i,n), где c=sqrt(gammaP/rho)в центрах ячеек полусумма

    # mask_speed_R3 = np.array(mask_speed_R3, dtype=bool)
    R_flux3 = np.zeros(n + 1)

    R_flux3[mask_speed_R3] = 0.5 * (np.roll(R_cons_half, 1)[mask_speed_R3[:n]] + R_cons_half[mask_speed_R3[:n]])

    if np.any(mask_speed_R3):

        G_R3 = 0.5 * (np.roll(G_cell, 1)[mask_speed_R3[:n]] + G_cell[mask_speed_R3[:n]])
    else:
        G_R3 = 0

    # характеристики сходятся - выполняем экстраполяцию
    # mask_speed_R4 = np.zeros(n + 1)
    # mask_speed_R4[1:n] = np.bitwise_and(R_speed_cell[1:n] < 0, np.roll(R_speed_cell, 1)[1:n] > 0)
    # mask_speed_R4 = np.array(mask_speed_R3, dtype=bool)
    #
    #
    # R_left4 = np.array(
    #     np.roll(U_flux, 1)[mask_speed_R4] + np.roll(G_cell, 1)[mask_speed_R4[:n]] * np.roll(P_flux, 1)[mask_speed_R4])
    #
    # R_right4 = U_flux[mask_speed_R4] + np.roll(G_cell, 1)[mask_speed_R4[:n]] * P_flux[mask_speed_R4]
    #
    # R_cent4 = np.array(
    #     np.roll(U_cons, 1)[mask_speed_R4[:n]] + np.roll(G_cell, 1)[mask_speed_R4[:n]] *
    #     np.roll(P_cons, 1)[mask_speed_R4[:n]]
    # )
    # R_flux4 = np.zeros(n + 1)
    # R_flux4[mask_speed_R4] = 0.5 * (R_left4 + R_right4)
    #
    # if np.any(mask_speed_R4):
    #     G_R4 = np.roll(G_cell, 1)[mask_speed_R4[:n]]
    # else:
    #     G_R4 = 0

    R_flux = R_flux12 + R_flux3 #+ R_flux4

    G_R = np.zeros(n - 1)
    G_R[mask_speed_R[1:n]] = G_R1
    G_R[mask_speed_R2[1:n]] = G_R2
    G_R[mask_speed_R3[1:n]] = G_R3
    # G_R[mask_speed_R3[1:n]] = G_R4


    # расчет инварианта Q
    speed = np.zeros(n + 1)
    speed[1:n] = np.roll(Q_speed_cell, 1)[1:n] * Q_speed_cell[1:n]
    mask_speed_Q = np.zeros(n + 1)
    mask_speed_Q[1:n] = np.bitwise_and(speed[1:n] > 0, Q_speed_cell[1:n] < 0)
    mask_speed_Q = np.array(mask_speed_Q, dtype=bool)

    min1 = np.zeros(n + 1)
    max1 = np.zeros(n + 1)

    # Q приходит справа
    Q_left = U_flux[mask_speed_Q] - G_cell[mask_speed_Q[:n]]
    Q_right = np.roll(U_flux, -1)[mask_speed_Q] - G_cell[mask_speed_Q[:n]]
    Q_cent = U_cons[mask_speed_Q[:n]] - G_cell[mask_speed_Q[:n]]

    Q_flux1 = np.zeros(n + 1)

    Q_flux1[mask_speed_Q] = (2 * Q_cons_half[mask_speed_Q[:n]] - (1 - dissip_Q) * Q_right) / (1 + dissip_Q)

    if np.any(mask_speed_Q):

        if monotize:
            Q_rh = RQ_right_monot * (
                    (Q_cons_half[mask_speed_Q[:n]] - Q_cent) / (tau * 0.5) + Q_speed_cell[mask_speed_Q[:n]] *
                    (Q_right - Q_left) / (np.roll(x, -1)[mask_speed_Q] - x[mask_speed_Q]))
            min2 = np.min(np.vstack([Q_left, Q_right, Q_cons_half[mask_speed_Q[:n]]]), axis=0) + tau * Q_rh
            max2 = np.max(np.vstack([Q_left, Q_right, Q_cons_half[mask_speed_Q[:n]]]), axis=0) + tau * Q_rh
            min1[mask_speed_Q] = min2
            max1[mask_speed_Q] = max2

        G_Q1 = G_cell[mask_speed_Q[:n]]

    else:
        G_Q1 = 0

    # G приходит слева

    mask_speed_Q2 = np.zeros(n + 1)
    mask_speed_Q2[1:n] = np.bitwise_and(speed[1:n] > 0, Q_speed_cell[1:n] > 0)
    mask_speed_Q2 = np.array(mask_speed_Q2, dtype=bool)

    Q_left2 = np.array(
        np.roll(U_flux, 1)[mask_speed_Q2] - np.roll(G_cell, 1)[mask_speed_Q2[:n]] * np.roll(P_flux, 1)[mask_speed_Q2])

    Q_right2 = U_flux[mask_speed_Q2] - np.roll(G_cell, 1)[mask_speed_Q2[:n]] * P_flux[mask_speed_Q2]
    Q_cent2 = np.array(
        np.roll(U_cons, 1)[mask_speed_Q2[:n]] - np.roll(G_cell, 1)[mask_speed_Q2[:n]] *
        np.roll(P_cons, 1)[mask_speed_Q2[:n]])

    Q_flux2 = np.zeros(n + 1)
    Q_flux2[mask_speed_Q2] = (2 * np.roll(Q_cons_half, 1)[mask_speed_Q2[:n]] -
                              (1 - dissip_Q) * Q_left2) / (1 + dissip_Q)

    if np.any(mask_speed_Q2):

        if monotize:
            Q_rh = RQ_right_monot * (
                    (np.roll(Q_cons_half, 1)[mask_speed_Q2[:n]] - Q_cent2) / (tau * 0.5) +
                    np.roll(Q_speed_cell, 1)[mask_speed_Q2[:n]] *
                    (Q_right2 - Q_left2) / (x[mask_speed_Q2] - np.roll(x, 1)[mask_speed_Q2]))

            min2 = np.min(np.vstack([Q_left2, Q_right2, np.roll(Q_cons_half, 1)[mask_speed_Q2[:n]]]),
                          axis=0) + tau * Q_rh
            max2 = np.max(np.vstack([Q_left2, Q_right2, np.roll(Q_cons_half, 1)[mask_speed_Q2[:n]]]),
                          axis=0) + tau * Q_rh

            min1[mask_speed_Q2] = min2
            max1[mask_speed_Q2] = max2

        G_Q2 = np.roll(G_cell, 1)[mask_speed_Q2[:n]]

    else:
        G_Q2 = 0

        # монотонизация решения
    Q_flux12 = Q_flux1 + Q_flux2

    if monotize:
        Q_flux_monot = np.bitwise_and(Q_flux12[1:n] <= max1[1:n], Q_flux12[1:n] >= min1[1:n]) * Q_flux12[1:n] + (
                Q_flux12[1:n] > max1[1:n]) * max1[1:n] + (Q_flux12[1:n] < min1[1:n]) * min1[1:n]
        Q_flux12[1:n] = Q_flux_monot

    # особая точка

    mask_speed_Q3 = np.zeros(n + 1)
    mask_speed_Q3[1:n] = (speed[1:n] < 0)
    mask_speed_Q3 = np.array(mask_speed_Q3, dtype=bool)

    Q_flux3 = np.zeros(n + 1)

    Q_flux3[mask_speed_Q3] = 0.5 * (np.roll(Q_cons_half, 1)[mask_speed_Q3[:n]] + Q_cons_half[mask_speed_Q3[:n]])

    if np.any(mask_speed_Q3):

        G_Q3 = 0.5 * (np.roll(G_cell, 1)[mask_speed_Q3[:n]] + G_cell[mask_speed_Q3[:n]])
    else:
        G_Q3 = 0

    Q_flux = Q_flux12 + Q_flux3
    G_Q = np.zeros(n - 1)
    G_Q[mask_speed_Q[1:n]] = G_Q1
    G_Q[mask_speed_Q2[1:n]] = G_Q2
    G_Q[mask_speed_Q3[1:n]] = G_Q3

    # Расчет U
    P_flux_new[1:n] = (R_flux[1:n] - Q_flux[1:n]) / (G_R + G_Q)
    U_flux_new[1:n] = Q_flux[1:n] + G_Q * P_flux_new[1:n]

    # Расчет инварианта S и инвариантов n - 1 компонент
    speed = U_flux_new[1:n]

    # условие, что если if np.abs(speed) < eps_u, то S приходит слева:
    mask_eps_u = np.zeros(n + 1)
    mask_eps_u[1:n] = (np.abs(speed) < eps_u)
    mask_eps_u = np.array(mask_eps_u, dtype=bool)
    C_used = np.zeros(n)

    S_flux1 = np.zeros(n + 1)
    S_flux1[mask_eps_u] = 0.5 * (np.roll(S_cons_half, 1)[mask_eps_u[:n]] + S_cons_half[mask_eps_u[:n]])
    C_used[mask_eps_u[:n]] = 0.5 * (np.roll(C_cell, 1)[mask_eps_u[:n]] + C_cell[mask_eps_u[:n]])
    Ksi_flux_new1 = np.zeros((n + 1, n_comp))
    Ksi_flux_new1[mask_eps_u, :n_comp - 1] = 0.5 * (
            np.roll(Ksi_cons_half, 1, axis=0)[mask_eps_u[:n], :n_comp - 1] +
            Ksi_cons_half[mask_eps_u[:n], :n_comp - 1])

    # условие  что если if np.abs(speed) > eps_u, то S приходит справа:
    mask_speed_S = np.zeros(n + 1)
    mask_speed_S[1:n] = np.bitwise_and(np.abs(speed) > eps_u, speed < 0)
    mask_speed_S = np.array(mask_speed_S, dtype=bool)

    S_left = P_flux[mask_speed_S] - C_cell[mask_speed_S[:n]] * C_cell[mask_speed_S[:n]] * Rho_flux[mask_speed_S]
    S_right = np.roll(P_flux, -1)[mask_speed_S] - C_cell[mask_speed_S[:n]] * C_cell[mask_speed_S[:n]] * \
                np.roll(Rho_flux, -1)[mask_speed_S]
    S_cent = P_cons[mask_speed_S[:n]] - C_cell[mask_speed_S[:n]] * C_cell[mask_speed_S[:n]] * Rho_cons[mask_speed_S[:n]]
    S_flux2 = np.zeros(n + 1)
    S_flux2[mask_speed_S] = (2 * S_cons_half[mask_speed_S[:n]] - (1 - dissip_S) * S_right) / (1 + dissip_S)

    # функция для монотонизации
    # def monotiz(Parametr_monot, cons_value, A_cent, A_left,A_right, mask_speed, tau, A_speed_cell, x, A_flux,n):
    #     Q_rh = Parametr_monot * (cons_value[mask_speed[:n]] - A_cent) / (tau * 0.5) + A_speed_cell[mask_speed[:n]] * (A_right - A_left) / (np.roll(x, -1)[mask_speed] - x[mask_speed])
    #     min2 = np.min(np.array([[A_left],[A_right], [cons_value[mask_speed[:n]]]]), axis = 0) + tau * Q_rh
    #     max2 = np.max(np.array([[A_left],[A_right], [cons_value[mask_speed[:n]]]]), axis = 0) + tau * Q_rh

    #     flux_monot = np.bitwise_and(A_flux[1:n] <= max2, A_flux[1:n] >= min2) * A_flux[1:n] + (A_flux[1:n] > max2) * max2 + (A_flux[1:n] < min2)* min2
    #     A_flux[1:n] = flux_monot
    #     return A_flux

    # if True in mask_speed_S:
    #     if monotize:
    #         S_flux23 = monotiz(S_right_monot,S_cons_half, S_cent,  S_left, S_right, mask_speed_S, tau, S_speed_cell, x, S_flux2,n)

    if np.any(mask_speed_S):
        if monotize:
            Q_rh = S_right_monot * (
                    (S_cons_half[mask_speed_S[:n]] - S_cent) / (tau * 0.5) + S_speed_cell[mask_speed_S[:n]] * (
                    S_right - S_left) / (np.roll(x, -1)[mask_speed_S] - x[mask_speed_S]))
            min2 = np.min(np.vstack([S_left, S_right, S_cons_half[mask_speed_S[:n]]]), axis=0) + tau * Q_rh

            max2 = np.max(np.vstack([S_left, S_right, S_cons_half[mask_speed_S[:n]]]), axis=0) + tau * Q_rh

            S_flux_monot = np.bitwise_and(S_flux2[mask_speed_S] <= max2, S_flux2[mask_speed_S] >= min2) * S_flux2[
                mask_speed_S] + (S_flux2[mask_speed_S] > max2) * max2 + (S_flux2[mask_speed_S] < min2) * min2
            S_flux2[mask_speed_S] = S_flux_monot

    S_left12 = Rho_comp_flux[mask_speed_S, :n_comp - 1] / Rho_flux.reshape(-1, 1)[mask_speed_S]
    S_right12 = np.roll(Rho_comp_flux, -1, axis=0)[mask_speed_S, :n_comp - 1] / \
                np.roll(Rho_flux, -1).reshape(-1, 1)[mask_speed_S]
    S_cent12 = Rho_comp_cons[mask_speed_S[:n], :n_comp - 1] / Rho_cons[mask_speed_S[:n]].reshape(-1, 1)
    Ksi_flux_new2 = np.zeros((n + 1, n_comp))
    Ksi_flux_new2[mask_speed_S, :n_comp - 1] = (2 * Ksi_cons_half[mask_speed_S[:n], :n_comp - 1]
                                                - (1 - dissip_Ksi) * S_right12) / (1 + dissip_Ksi)

    if np.any(mask_speed_S):
        if monotize:
            Q_rh = Ksi_right_monot * ((Ksi_cons_half[mask_speed_S[:n], :n_comp - 1] - S_cent12) / (tau * 0.5) +
                                      S_speed_cell.reshape(-1, 1)[mask_speed_S[:n]] * (S_right12 - S_left12) / (
                                    np.roll(x, -1).reshape(-1, 1)[mask_speed_S] - x.reshape(-1, 1)[mask_speed_S]))

            min2 = np.min(np.vstack([S_left12, S_right12, Ksi_cons_half[mask_speed_S[:n], :n_comp - 1]]),
                          axis=0) + tau * Q_rh

            max2 = np.max(np.vstack([S_left12, S_right12, Ksi_cons_half[mask_speed_S[:n], :n_comp - 1]]),
                          axis=0) + tau * Q_rh

            Ksi_flux_monot = np.bitwise_and(Ksi_flux_new2[mask_speed_S, :n_comp - 1] <= max2,
                                            Ksi_flux_new2[mask_speed_S, :n_comp - 1] >= min2) * \
                                            Ksi_flux_new2[mask_speed_S,:n_comp - 1] + \
                                            (Ksi_flux_new2[mask_speed_S, :n_comp - 1] > max2) * max2 + \
                                            (Ksi_flux_new2[mask_speed_S, :n_comp - 1] < min2) * min2

            Ksi_flux_new2[mask_speed_S, :n_comp - 1] = Ksi_flux_monot

    C_used[mask_speed_S[:n]] = C_cell[mask_speed_S[:n]]

    #  S приходит слева
    # условие  что если if np.abs(speed) > eps_u, то S приходит слева:
    mask_speed_S2 = np.zeros(n + 1)
    mask_speed_S2[1:n] = np.bitwise_and(np.abs(speed) > eps_u, speed > 0)
    mask_speed_S2 = np.array(mask_speed_S2, dtype=bool)

    S_left = np.roll(P_flux, 1)[mask_speed_S2] - np.roll(C_cell, 1)[mask_speed_S2[:n]] * \
             np.roll(C_cell, 1)[mask_speed_S2[:n]] * np.roll(Rho_flux, 1)[mask_speed_S2]

    S_right = P_flux[mask_speed_S2] - np.roll(C_cell, 1)[mask_speed_S2[:n]] * \
              np.roll(C_cell, 1)[mask_speed_S2[:n]] * Rho_flux[mask_speed_S2]

    S_cent = np.roll(P_cons, 1)[mask_speed_S2[:n]] - np.roll(C_cell, 1)[mask_speed_S2[:n]] *\
             np.roll(C_cell, 1)[mask_speed_S2[:n]] * np.roll(Rho_cons, 1)[mask_speed_S2[:n]]

    S_flux3 = np.zeros(n + 1)
    S_flux3[mask_speed_S2] = (2 * np.roll(S_cons_half, 1)[mask_speed_S2[:n]] - (1 - dissip_S) * S_left) / (1 + dissip_S)

    if np.any(mask_speed_S2):

        if monotize:
            Q_rh = S_right_monot * ((np.roll(S_cons_half, 1)[mask_speed_S2[:n]] - S_cent) / (tau * 0.5) + \
                                    np.roll(S_speed_cell, 1)[mask_speed_S2[:n]] *
                                    (S_right - S_left) / (x[mask_speed_S2] - np.roll(x, 1)[mask_speed_S2]))

            min2 = np.min(np.vstack([S_left, S_right, np.roll(S_cons_half, 1)[mask_speed_S2[:n]]]), axis=0) + tau * Q_rh
            max2 = np.max(np.vstack([S_left, S_right, np.roll(S_cons_half, 1)[mask_speed_S2[:n]]]), axis=0) + tau * Q_rh

            S_flux_monot = np.bitwise_and(S_flux3[mask_speed_S2] <= max2, S_flux3[mask_speed_S2] >= min2) *  \
                                          S_flux3[mask_speed_S2] + (S_flux3[mask_speed_S2] > max2) * max2 + \
                                          (S_flux3[mask_speed_S2] < min2) * min2

            S_flux3[mask_speed_S2] = S_flux_monot

    S_left12 = np.roll(Rho_comp_flux, 1, axis=0)[mask_speed_S2, :n_comp - 1] / \
               np.roll(Rho_flux, 1).reshape(-1, 1)[mask_speed_S2]

    S_right12 = Rho_comp_flux[mask_speed_S2, : n_comp - 1] / Rho_flux.reshape(-1, 1)[mask_speed_S2]

    S_cent12 = np.roll(Rho_comp_cons, 1, axis=0)[mask_speed_S2[:n], :n_comp - 1] / \
               np.roll(Rho_cons, 1)[mask_speed_S2[:n]].reshape(-1, 1)

    Ksi_flux_new3 = np.zeros((n + 1, n_comp))
    Ksi_flux_new3[mask_speed_S2, :n_comp - 1] = (2 * np.roll(Ksi_cons_half, 1, axis=0)[mask_speed_S2[:n], :n_comp - 1] -
                                                 (1 - dissip_Ksi) * S_left12) / (1 + dissip_Ksi)

    if np.any(mask_speed_S2):
        if monotize:
            Q_rh = Ksi_right_monot * (
                    (np.roll(Ksi_cons_half, 1, axis=0)[mask_speed_S2[:n], :n_comp - 1] - S_cent12) / (tau * 0.5) +
                    np.roll(S_speed_cell, 1).reshape(-1, 1)[mask_speed_S2[:n]] * (S_right12 - S_left12) / \
                    (np.roll(x, -1).reshape(-1, 1)[mask_speed_S2] - x.reshape(-1, 1)[mask_speed_S2]))

            min2 = np.min(
                np.vstack([S_left12, S_right12, np.roll(Ksi_cons_half, 1, axis=0)[mask_speed_S2[:n], :n_comp - 1]]),
                axis=0) + tau * Q_rh

            max2 = np.max(
                np.vstack([S_left12, S_right12, np.roll(Ksi_cons_half, 1, axis=0)[mask_speed_S2[:n], :n_comp - 1]]),
                axis=0) + tau * Q_rh

            Ksi_flux_monot = np.bitwise_and(Ksi_flux_new3[mask_speed_S2, :n_comp - 1] <= max2,
                                            Ksi_flux_new3[mask_speed_S2, :n_comp - 1] >= min2) *\
                                            Ksi_flux_new3[mask_speed_S2, :n_comp - 1] + \
                                            (Ksi_flux_new3[mask_speed_S2, :n_comp - 1] > max2) * max2 + \
                                            (Ksi_flux_new3[mask_speed_S2, :n_comp - 1] < min2) * min2
            Ksi_flux_new3[mask_speed_S2, :n_comp - 1] = Ksi_flux_monot

    C_used[mask_speed_S2[:n]] = np.roll(C_cell, 1)[mask_speed_S2[:n]]

    S_flux = S_flux1 + S_flux2 + S_flux3

    Ksi_flux_new = Ksi_flux_new1 + Ksi_flux_new2 + Ksi_flux_new3

    # Вычисляем последнюю компоненту
    Ksi_flux_new[1:n, n_comp - 1] = 1.0
    Ksi_flux_new[1:n, n_comp - 1] -= np.sum(Ksi_flux_new[1:n, :n_comp - 1], axis=1)

    Rho_flux_new[1:n] = (P_flux_new[1:n] - S_flux[1:n]) / (C_used[1:n] * C_used[1:n])

    # ------------------------------------------------------------------------------------------------------------------
    # Граничные условия типа p = p0 или v = v0 на обоих концах
    # Левая граница
    # Q приходит справа
    # ------------------------------------------------------------------------------------------------------------------

    Q_left = U_flux[0] - G_cell[0] * P_flux[0]
    Q_right = U_flux[1] - G_cell[0] * P_flux[1]
    Q_cent = U_cons[0] - G_cell[0] * P_cons[0]
    Q_flux[0] = (2 * Q_cons_half[0] - (1 - dissip_Q) * Q_right) / (1 + dissip_Q)

    if monotize:
        Q_rh = RQ_right_monot * ((Q_cons_half[0] - Q_cent) / (tau * 0.5) + Q_speed_cell[0] * \
                                 (Q_right - Q_left) / (x[1] - x[0]))
        min1 = min(min(Q_left, Q_right), Q_cons_half[0]) + tau * Q_rh
        max1 = max(max(Q_left, Q_right), Q_cons_half[0]) + tau * Q_rh
        if Q_flux[0] > max1:
            Q_flux[0] = max1
        else:
            if Q_flux[0] < min1:
                Q_flux[0] = min1
    G_Q = G_cell[0]

    if bc_left_p:
        P_flux_new[0] = P_left
        U_flux_new[0] = Q_flux[0] + G_cell[0] * P_flux_new[0]

    if bc_left_v:
        U_flux_new[0] = U_left
        P_flux_new[0] = (U_flux_new[0] - Q_flux[0]) / G_cell[0]

    if bc_left_R:
        R_flux[0] = R_left_bc
        P_flux_new[0] = ((R_flux[0] - Q_flux[0]) / (G_left + G_cell[0]))
        U_flux_new[0] = Q_flux[0] + G_cell[0] * P_flux_new[0]

    if np.abs(U_flux_new[0]) < eps_u:
        if bc_left_R:
            S_flux[0] = 0.5 * (S_left_bc + S_cons_half[0])
            C_used1 = 0.5 * (C_left + C_cell[0])
            Rho_flux_new[0] = (P_flux_new[0] - S_flux[0]) / (C_used1 * C_used1)
        else:
            S_flux[0] = S_cons_half[0]
            Rho_flux_new[0] = (P_flux_new[0] - S_flux[0]) / (C_cell[0] * C_cell[0])

        Ksi_flux_new[0, n_comp - 1] = 1.0

        for j in range(n_comp - 1):
            Ksi_flux_new[0, j] = Ksi_cons_half[0, j]
            Ksi_flux_new[0, n_comp - 1] -= Ksi_flux_new[0, j]

    else:
        if U_flux_new[0] < 0:
            # S приходит справа
            S_left = P_flux[0] - C_cell[0] * C_cell[0] * Rho_flux[0]
            S_right = P_flux[1] - C_cell[0] * C_cell[0] * Rho_flux[1]
            S_cent = P_cons[0] - C_cell[0] * C_cell[0] * Rho_cons[0]
            S_flux[0] = (2 * S_cons_half[0] - (1 - dissip_S) * S_right) / (1 + dissip_S)

            if monotize:
                Q_rh = S_right_monot * (
                        (S_cons_half[0] - S_cent) / (tau * 0.5) + S_speed_cell[0] * (S_right - S_left) / (
                        x[1] - x[0]))
                min1 = min(min(S_left, S_right), S_cons_half[0]) + tau * Q_rh
                max1 = max(max(S_left, S_right), S_cons_half[0]) + tau * Q_rh
                if S_flux[0] > max1:
                    S_flux[0] = max1
                else:
                    if S_flux[0] < min1:
                        S_flux[0] = min1

            for j in range(n_comp - 1):
                S_left = Rho_comp_flux[0, j] / Rho_flux[0]
                S_right = Rho_comp_flux[1, j] / Rho_flux[1]
                S_cent = Rho_comp_cons[0, j] / Rho_cons[0]
                Ksi_flux_new[0, j] = (2 * Ksi_cons_half[0, j] - (1 - dissip_Ksi) * S_right) / (1 + dissip_Ksi)

                if monotize:
                    Q_rh = Ksi_right_monot * (
                            (Ksi_cons_half[0, j] - S_cent) / (tau * 0.5) + S_speed_cell[0] * (S_right - S_left) / (
                            x[1] - x[0]))
                    min1 = min(min(S_left, S_right), Ksi_cons_half[0, j]) + tau * Q_rh
                    max1 = max(max(S_left, S_right), Ksi_cons_half[0, j]) + tau * Q_rh
                    if Ksi_flux_new[0, j] > max1:
                        Ksi_flux_new[0, j] = max1
                    else:
                        if Ksi_flux_new[0, j] < min1:
                            Ksi_flux_new[0, j] = min1

            Ksi_flux_new[0, n_comp - 1] = 1.0
            for j in range(n_comp - 1):
                Ksi_flux_new[0, n_comp - 1] -= Ksi_flux_new[0, j]

        else:
            if bc_left_R:
                S_flux[0] = S_left_bc
                Rho_flux_new[0] = (P_flux_new[0] - S_flux[0]) / (C_left * C_left)

            else:
                S_flux[0] = S_cons_half[0]
                Rho_flux_new[0] = (P_flux_new[0] - S_flux[0]) / (C_cell[0] * C_cell[0])

            Ksi_flux_new[0, n_comp - 1] = 1.0
            for j in range(n_comp - 1):
                Ksi_flux_new[0, j] = Ksi_cons_half[0, j]
                Ksi_flux_new[0, n_comp - 1] -= Ksi_flux_new[0, j]
    # ------------------------------------------------------------------------------------------------------------------
    # Правая граница
    # R приходит слева
    # ------------------------------------------------------------------------------------------------------------------

    R_left = U_flux[n - 1] + G_cell[n - 1] * P_flux[n - 1]
    R_right = U_flux[n] + G_cell[n - 1] * P_flux[n]
    R_cent = U_cons[n - 1] + G_cell[n - 1] * P_cons[n - 1]
    R_flux[n] = (2 * R_cons_half[n - 1] - (1 - dissip_R) * R_left) / (1 + dissip_R)

    if monotize:
        Q_rh = RQ_right_monot * ((R_cons_half[n - 1] - R_cent) / (tau * 0.5) + R_speed_cell[n - 1] *
                                 (R_right - R_left) / (x[n] - x[n - 1]))
        min1 = min(min(R_left, R_right), R_cons_half[n - 1]) + tau * Q_rh
        max1 = max(max(R_left, R_right), R_cons_half[n - 1]) + tau * Q_rh
        if R_flux[n] > max1:
            R_flux[n] = max1
        else:
            if R_flux[n] < min1:
                R_flux[n] = min1
    G_R = G_cell[n - 1]

    if bc_right_p:
        P_flux_new[n] = P_right
        U_flux_new[n] = R_flux[n] - G_cell[n - 1] * P_flux_new[n]

    if bc_right_v:
        U_flux_new[n] = U_right
        P_flux_new[n] = (R_flux[n] - U_flux_new[n]) / G_cell[n - 1]

    if bc_right_Q:
        Q_flux[n] = Q_right_bc
        P_flux_new[n] = (R_flux[n] - Q_flux[n]) / (G_cell[n - 1] + G_right)
        U_flux_new[n] = Q_flux[n] + G_right * P_flux_new[n]

    if np.abs(U_flux_new[n]) < eps_u:
        if bc_right_Q:
            S_flux[n] = 0.5 * (S_right_bc + S_cons_half[n - 1])
            C_used2 = 0.5 * (C_right + C_cell[n - 1])
            Rho_flux_new[n] = (P_flux_new[n] - S_flux[n]) / (C_used2 * C_used2)

        else:
            S_flux[n] = S_cons_half[n - 1]
            Rho_flux_new[n] = (P_flux_new[n] - S_flux[n]) / (C_cell[n - 1] * C_cell[n - 1])

        Ksi_flux_new[n, n_comp - 1] = 1.0

        for j in range(n_comp - 1):
            Ksi_flux_new[n, j] = Ksi_cons_half[n - 1, j]
            Ksi_flux_new[n, n_comp - 1] -= Ksi_flux_new[n, j]

    else:
        if U_flux_new[n - 1] > 0:
            # S приходит слева
            S_left = P_flux[n - 1] - C_cell[n - 1] * C_cell[n - 1] * Rho_flux[n - 1]
            S_right = P_flux[n] - C_cell[n - 1] * C_cell[n - 1] * Rho_flux[n]
            S_cent = P_cons[n - 1] - C_cell[n - 1] * C_cell[n - 1] * Rho_cons[n - 1]

            S_flux[n] = (2 * S_cons_half[n - 1] - (1 - dissip_S) * S_left) / (1 + dissip_S)

            if monotize:
                Q_rh = S_right_monot * ((S_cons_half[n - 1] - S_cent) / (tau * 0.5) + S_speed_cell[n - 1] *
                                        (S_right - S_left) / (x[n] - x[n - 1]))
                min1 = min(min(S_left, S_right), S_cons_half[n - 1]) + tau * Q_rh
                max1 = max(max(S_left, S_right), S_cons_half[n - 1]) + tau * Q_rh
                if S_flux[n] > max1:
                    S_flux[n] = max1
                else:
                    if S_flux[n] < min1:
                        S_flux[n] = min1
            Rho_flux_new[n] = (P_flux_new[n] - S_flux[n]) / (C_cell[n - 1] * C_cell[n - 1])

            for j in range(n_comp - 1):
                S_left = Rho_comp_flux[n - 1, j] / Rho_flux[n - 1]
                S_right = Rho_comp_flux[n, j] / Rho_flux[n]
                S_cent = Rho_comp_cons[n - 1, j] / Rho_cons[n - 1]
                Ksi_flux_new[n, j] = (2 * Ksi_cons_half[n - 1, j] - (1 - dissip_Ksi) * S_left) / (1 + dissip_Ksi)

                if monotize:
                    Q_rh = Ksi_right_monot * ((Ksi_cons_half[n - 1, j] - S_cent) / (tau * 0.5) +
                                              S_speed_cell[n - 1] * (S_right - S_left) / (x[n] - x[n - 1]))
                    min1 = min(min(S_left, S_right), Ksi_cons_half[n - 1, j]) + tau * Q_rh
                    max1 = max(max(S_left, S_right), Ksi_cons_half[n - 1, j]) + tau * Q_rh
                    if Ksi_flux_new[n, j] > max1:
                        Ksi_flux_new[n, j] = max1
                    else:
                        if Ksi_flux_new[n, j] < min1:
                            Ksi_flux_new[n, j] = min1

            Ksi_flux_new[n, n_comp - 1] = 1.0
            for j in range(n_comp - 1):
                Ksi_flux_new[n, n_comp - 1] -= Ksi_flux_new[n, j]

        else:
            if bc_right_Q:
                S_flux[n] = S_right_bc
                Rho_flux_new[n] = (P_flux_new[n] - S_flux[n]) / (C_right * C_right)

            else:
                S_flux[n] = S_cons_half[n - 1]

                Rho_flux_new[n] = (P_flux_new[n] - S_flux[n]) / (C_cell[n - 1] * C_cell[n - 1]);

            Ksi_flux_new[n, n_comp - 1] = 1.0
            for j in range(n_comp - 1):
                Ksi_flux_new[n, j] = Ksi_cons_half[n - 1, j]
                Ksi_flux_new[n, n_comp - 1] -= Ksi_flux_new[n, j]

    # Забываем старые потоковые инварианты и вычисляем плотности компонент
    Rho_comp_flux_tmp = np.zeros(((n + 1), n_comp))

    U_flux = U_flux_new
    P_flux = P_flux_new
    Rho_flux = Rho_flux_new
    up = 0
    down = 0

    for j in range(n_comp):
        Rho_comp_flux[:, j] = Ksi_flux_new[:, j] * Rho_flux[:];
        up += gamma[j] * c_temp[j] * Ksi_flux_new[:, j];
        down += c_temp[j] * Ksi_flux_new[:, j];
        Ksi_comp_flux[:, j] = Ksi_flux_new[:, j];

    gamma_loc_flux = up / down

    E_flux = 0.5 * U_flux * U_flux + P_flux / (Rho_flux * (gamma_loc_flux - 1))

    # третья конcервативная фаза алгоритма
    # Фаза 1 + вычисление консервативных инвариантов для монотонизации

    Rho_cons = np.zeros(n)
    Rho_comp_cons = Rho_comp_cons_half - 0.5 * tau * (
            np.roll(U_flux.reshape(-1, 1), -1)[:n] * np.roll(Rho_comp_flux, -1, axis=0)[:n] - \
            U_flux.reshape(-1, 1)[:n] * Rho_comp_flux[:n]) / (np.roll(x, -1)[:n] - x[:n]).reshape(-1, 1)
    Rho_cons = np.sum(Rho_comp_cons, axis=1)

    U_cons = Rho_cons_half * U_cons_half - 0.5 * tau * (np.roll(Rho_flux, -1)[:n] * np.roll(U_flux, -1)[:n] ** 2 \
                                                        + np.roll(P_flux, -1)[:n] - Rho_flux[:n] * U_flux[
                                                                                                   :n] ** 2 - P_flux[
                                                                                                              :n]) / (
                     np.roll(x, -1)[:n] - x[:n])
    U_cons /= Rho_cons
    E_cons = Rho_cons_half * E_cons_half - 0.5 * tau * (np.roll(Rho_flux, -1)[:n] *
                                                        np.roll(U_flux, -1)[:n] *
                                                        np.roll(E_flux, -1)[:n] +
                                                        np.roll(P_flux, -1)[:n] *
                                                        np.roll(U_flux, -1)[:n] -
                                                        Rho_flux[:n] * U_flux[:n] *
                                                        E_flux[:n] - P_flux[:n] *
                                                        U_flux[:n]) / (np.roll(x, -1)[:n] - x[:n])
    E_cons /= Rho_cons
    up = np.sum(gamma * c_temp * Rho_comp_cons, axis=1)
    down = np.sum(c_temp * Rho_comp_cons, axis=1)
    gamma_loc_cons = up / down
    P_cons = (E_cons - 0.5 * U_cons * U_cons) * Rho_cons * (gamma_loc_cons - 1)

    # l_time = time.as_integer_ratio()
    # расчет нового шага по времени

    n_step += 1
    if n_step % Cour_incr_freq == 0 and Cour <= 0.15:
        Cour *= 2.0

    min_tau_0 = np.inf

    min_tau_1 = np.min(Cour * h / (np.sqrt(gamma_loc_cons * P_cons / Rho_cons) + np.abs(U_cons)))

    flux = (np.roll(Rho_comp_flux, -1, axis=0)[:n] * np.roll(U_flux.reshape(-1, 1), -1)[:n] -
            Rho_comp_flux[:n] * U_flux.reshape(-1, 1)[:n]) / (np.roll(x, -1)[:n] - x[:n]).reshape(-1, 1)

    idx = np.bitwise_and(flux > 0, Rho_comp_cons > 0)

    tau_tmp = Rho_comp_cons[idx] / flux[idx]

    if np.any(tau_tmp):
        min_tau_2 = np.min(tau_tmp)
    else:
        min_tau_2 = np.inf

    flux1 = (np.roll(Rho_flux, -1)[:n] * np.roll(U_flux, -1)[:n] * np.roll(U_flux, -1)[:n] -
             Rho_flux[:n] * U_flux[:n] * U_flux[:n]) / (np.roll(x, -1)[:n] - x[:n])

    indx = np.bitwise_and(np.abs(flux1) > eps_u, np.abs(Rho_cons * U_cons) > eps_u)
    tau_tmp2 = np.abs(Rho_cons[indx] * U_cons[indx] / flux1[indx])

    if np.any(tau_tmp2):
        min_tau_3 = np.min(tau_tmp2)
    else:
        min_tau_3 = np.inf

    min_tau = min(min_tau_0, min_tau_1, min_tau_2, min_tau_3)

    tau = min_tau
    # t_steps.append(tm.perf_counter() - t_step_start)
    # print(time)

print(f"{tm.perf_counter() - t_start}, {np.mean(t_steps)}")

# plt.figure(figsize=(15, 10), dpi=200)
plt.grid(True)

plt.title(f' схема Кабаре, {n} точек')
plt.plot(x[:-1], Rho_cons, marker='>', color='g')

# plt.savefig("Кабарe")
plt.show()
