import numpy as np
import pandas as pd
import warnings
import math


warnings.simplefilter("ignore")


df = pd.read_csv("WineDataset.csv")
features = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
            "Proline", "Wine"]


def train_test_split(df, x_cols, y_col, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    df_copy = df.sample(frac=1).reset_index(drop=True)
    train_size = math.ceil(df_copy.shape[0] * (1 - test_size))
    return (df_copy[x_cols][:train_size], df_copy[x_cols][train_size:], df_copy[y_col][:train_size],
            df_copy[y_col][train_size:])


def dist_sq(p1, p2):
    return sum((i - j) ** 2 for i, j in zip(p1, p2))


def get_class_by_distances(distances, k):
    classes = [i[1] for i in distances[:k]]
    return max(classes, key=classes.count)


def get_error_matrix(y_true, y_pred):
    classes = [1, 2, 3]
    matrix = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
    for i, pred_class in enumerate(classes):
        for j, actual_class in enumerate(classes):
            matrix[i][j] = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t == actual_class and y_p == pred_class)
    return matrix


def print_error_matrix_pretty(matrix):
    df_error = pd.DataFrame(matrix, columns=["True 1", "True 2", "True 3"],
                            index=["Predicted 1", "Predicted 2", "Predicted 3"])
    print(df_error)
    accuracy_all = sum(matrix[i][i] for i in range(len(matrix))) / sum(sum(i) for i in matrix)
    print(f"Accuracy among all: {round(accuracy_all, 3)}")
    print()


def knn(df, x_columns, y_column, k, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(df, x_columns, y_column, test_size, random_state)
    x_train_numpy = x_train.to_numpy()
    y_train_numpy = y_train.to_numpy()
    y_pred = np.zeros(y_test.shape)
    for idx, new_val in enumerate(x_test.to_numpy()):
        distances = [(dist_sq(val, new_val), cl) for val, cl in zip(x_train_numpy, y_train_numpy)]
        distances.sort(key=lambda val: val[0])
        y_pred[idx] = get_class_by_distances(distances, k)
    return get_error_matrix(y_test, y_pred)


y_col = features[-1]
# set1
x_cols = features[:-1]
print_error_matrix_pretty(knn(df, x_cols, y_col, 3))
print_error_matrix_pretty(knn(df, x_cols, y_col, 7))
print_error_matrix_pretty(knn(df, x_cols, y_col, 15))
