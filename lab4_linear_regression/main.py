import pandas as pd
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


df = pd.read_csv("california_housing_train.csv")
features = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income", "median_house_value"]

for feature in features:
    plt.figure(figsize=(10, 10))
    plt.hist(df[feature], bins=100)
    plt.title(f"Visual representation of {feature}\n")
    plt.xlabel(feature)
    plt.ylabel("Number of data entities with this value")

    quantiles = df[feature].quantile([0.25, 0.5, 0.75])
    for quantile, quantile_value in quantiles.items():
        plt.axvline(x=quantile_value, color="red", linestyle=":", label=f"{quantile*100}% квантиль")
    plt.axvline(x=df[feature].mean(), color="green", linestyle=":", label=f"Среднее значение {int(df[feature].mean() * 1000) / 1000}")

    print(f"Statistics for feature {feature}\n")
    print(df[features].describe()[feature].to_string())

    plt.legend()
    # plt.show()


# Minmax normalization
def normalize(col: pd.Series):
    col_min = col.min()
    col_max = col.max()
    for index, value in col.items():
        col[index] = (col[index] - col_min) / (col_max - col_min)


def train_test_split(df, x_cols, y_col, test_size=0.2):
    df_copy = df.sample(frac=1).reset_index(drop=True)
    train_size = math.ceil(df_copy.shape[0] * (1 - test_size))
    return (df_copy[x_cols][:train_size], df_copy[x_cols][train_size:], df_copy[y_col][:train_size],
            df_copy[y_col][train_size:])


df4 = df.copy()
df.apply(normalize, axis=0)
df4["mean_households"] = df4["households"] / df4["population"]
df4.apply(normalize, axis=0)


def col_mult_sum(x_train_numpy, col_num1, col_num2):
    return sum(i[col_num1] * i[col_num2] for i in x_train_numpy)


def swap_col(x_train_numpy, col_num, new_col):
    for i in range(x_train_numpy.shape[0]):
        x_train_numpy[i][col_num] = new_col[i]


def linear_regression(df, x_cols, y_col, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(df, x_cols, y_col, test_size)
    x_train.insert(0, "Extra", np.ones(x_train.shape[0]))
    x_train_numpy = x_train.to_numpy()
    A = np.zeros((x_train_numpy.shape[1], x_train_numpy.shape[1]))
    B = np.zeros((x_train_numpy.shape[1], 1))
    y_train_numpy = y_train.to_numpy()
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
    return coefs, x_test, y_test


def r_square(y_expected, y_ans):
    mean_y = y_expected.mean()
    return 1 - (sum((y_ans[i] - y_expected[i]) ** 2 for i in range(y_ans.shape[0])) / sum((y_expected[i] - mean_y) ** 2 for i in range(y_ans.shape[0])))


def calculate(coefs, xs):
    return np.array([sum(coefs[i + 1] * xs[j][i] for i in range(xs.shape[1])) + coefs[0] for j in range(xs.shape[0])])


def get_r_square(coefs, x_test, y_test):
    return r_square(y_test.to_numpy(), calculate(coefs, x_test.to_numpy()))


def features_without_cols(columns_to_remove):
    cur_features = features[:-1].copy()
    for col in columns_to_remove:
        cur_features.remove(col)
    return cur_features


# common for all sets
y_column = features[-1]
# set 1
x_columns_1 = features[:-1]
coefs_1, x_test_1, y_test_1 = linear_regression(df, x_columns_1, y_column)
print("set1 R^2:", round(get_r_square(coefs_1, x_test_1, y_test_1), 5))

# set 2
x_columns_2 = features_without_cols(["median_income"])
coefs_2, x_test_2, y_test_2 = linear_regression(df, x_columns_2, y_column)
print("set2 R^2:", round(get_r_square(coefs_2, x_test_2, y_test_2), 5))

# set 3
x_columns_3 = features_without_cols(["households"])
coefs_3, x_test_3, y_test_3 = linear_regression(df, x_columns_3, y_column)
print("set3 R^2:", round(get_r_square(coefs_3, x_test_3, y_test_3), 5))

# set 4
x_columns_4 = features_without_cols(["households", "population"])
x_columns_4.append("mean_households")
coefs_4, x_test_4, y_test_4 = linear_regression(df4, x_columns_4, y_column)
print("set4 R^2:", round(get_r_square(coefs_4, x_test_4, y_test_4), 5))

