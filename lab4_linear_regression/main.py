import pandas as pd
import numpy as np
import math

df = pd.read_csv("california_housing_train.csv")
features = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income", "median_house_value"]

# print(df.count())
# print(df.mean())
# print(df.std())
# print(df.min())
# print(df.max())
# TODO квантили
# TODO визуализация


# Minmax normalization
def normalize(col: pd.Series):
    col_min = col.min()
    col_max = col.max()
    for index, value in col.items():
        col[index] = (col[index] - col_min) / (col_max - col_min)


def train_test_split(df, x_cols, y_col, test_size, random_state):
    df_copy = df.copy()
    np.random.seed(random_state)
    np.random.shuffle(df_copy.values)
    train_size = math.ceil(df_copy.shape[0] * test_size)
    return df_copy[x_cols][:train_size], df_copy[x_cols][train_size:], df_copy[y_col][:train_size], df_copy[y_col][train_size:]


df.apply(normalize, axis=0)
x_columns = features[:-1]
y_column = features[-1]
x_train, x_test, y_train, y_test = train_test_split(df, x_columns, y_column, test_size=0.2, random_state=42)
matrix = []
x_train.insert(0, "Extra", np.ones(x_train.shape[0]))
x_train_numpy = x_train.to_numpy()
A = np.zeros((x_train_numpy.shape[1], x_train_numpy.shape[1]))
B = np.zeros((x_train_numpy.shape[1], 1))
y_train_numpy = y_train.to_numpy()


def col_mult_sum(x_train_numpy, col_num1, col_num2):
    return sum(i[col_num1] * i[col_num2] for i in x_train_numpy)


def swap_col(x_train_numpy, col_num, new_col):
    for i in range(x_train_numpy.shape[0]):
        x_train_numpy[i][col_num] = new_col[i]


for i in range(x_train_numpy.shape[1]):
    B[i] = sum(y_train_numpy[j] * x_train_numpy[j][i] for j in range(x_train_numpy.shape[0]))
    for j in range(x_train_numpy.shape[1]):
        A[i][j] = col_mult_sum(x_train_numpy, j, i)


det = np.linalg.det(A)
other_dets = []
for i in range(A.shape[1]):
    cur_col = [A[j][i] for j in range(A.shape[0])]
    swap_col(A, i, B)
    other_dets += [np.linalg.det(A)]
    swap_col(A, i, cur_col)

coefs = [i / det for i in other_dets]


def r_square(y_expected, y_ans):
    mean_y = y_expected.mean()
    return 1 - (sum((y_ans[i] - y_expected[i]) ** 2 for i in range(y_ans.shape[0])) / sum((y_expected[i] - mean_y) ** 2 for i in range(y_ans.shape[0])))


def calculate(coefs, xs):
    return np.array([sum(coefs[i + 1] * xs[j][i] for i in range(xs.shape[1])) + coefs[0] for j in range(xs.shape[0])])


y_ans = calculate(coefs, x_test.to_numpy())
print("R^2:", r_square(y_test.to_numpy(), y_ans))
