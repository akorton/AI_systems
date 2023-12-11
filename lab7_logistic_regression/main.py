import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class MyLogReg:

    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None, learning_rate_func=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.error_matrix = None
        self.tpr = None
        self.fpr = None
        self.learning_rate_func = learning_rate_func

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=0):
        curX = X.copy()
        curX.insert(0, "dummy", np.ones(curX.shape[0]))
        self.weights = np.ones(curX.shape[1])
        y_pred = []
        for i in range(self.n_iter):
            if self.learning_rate_func:
                self.learning_rate = self.learning_rate_func(i)
            y_pred = self.predict_proba(curX, False)
            self.error_matrix = self.get_error_matrix(y, self.predict(curX))
            if verbose and (i + 1) % verbose == 0:
                log_loss = self.log_loss(y, y_pred)
                log_loss = int(log_loss*1000)/1000
                print(f"[debug] iter={i+1} log_loss={log_loss} learning_rate={int(self.learning_rate * 100000) / 100000}")
            grad = self.gradient(y, y_pred, curX)
            self.weights -= self.learning_rate*grad
        self.generate_roc_data(y, y_pred)

    def generate_roc_data(self, y_true, y_pred, total_points=10000):
        self.tpr = []
        self.fpr = []
        for eps in range(total_points+1):
            threshold = 1 - eps / total_points
            cur_pred = [1 if i > threshold else 0 for i in y_pred]
            matrix = self.get_error_matrix(y_true, cur_pred)
            self.tpr.append(matrix[0][0] / sum(matrix[0]))
            self.fpr.append(matrix[1][0] / sum(matrix[1]))
        self.fpr.append(1)
        self.tpr.append(self.tpr[-1])

    def integral(self, x, y):
        ans = 0
        for i in range(len(x) - 1):
            ans += ((y[i] + y[i+1])/2) * (x[i + 1] - x[i])
        return ans

    def predict(self, X, threshold=0.5, add_dummy_col=False):
        curX = X.copy()
        if add_dummy_col:
            curX.insert(0, "dummy", np.ones(X.shape[0]))
        classify = lambda x: 1 if x > threshold else 0
        return [classify(i) for i in self.sigmoid(np.dot(curX.values, self.weights))]

    def predict_proba(self, X, add_dummy_col=True):
        curX = X.copy()
        if add_dummy_col:
            curX.insert(0, "dummy", np.ones(X.shape[0]))
        return self.sigmoid(np.dot(curX.values, self.weights))

    def sigmoid(self, val):
        return 1 / (1 + np.e**(-val))

    def get_coef(self):
        return self.weights[1:]

    def log_loss(self, y_true, y_pred, eps=1e-15):
        return -1/len(y_true)*sum(i*np.log(j+eps) + (1 - i)*np.log(1 - j + eps) for i,j in zip(y_true, y_pred))

    def gradient(self, y_true, y_pred, X):
        return 1/len(y_true)*np.dot(y_pred - y_true, X)

    def get_error_matrix(self, y_true, y_pred):
        matrix = []
        classes = [1, 0]
        for real in classes:
            cur = []
            for pred in classes:
                cur.append(sum(1 if i == real and j == pred else 0 for i, j in zip(y_true, y_pred)))
            matrix.append(cur)
        return matrix

    def get_best_score(self):
        ans = 0
        elements_num = sum(sum(i) for i in self.error_matrix)
        try:
            if self.metric == "accuracy":
                ans = (self.error_matrix[0][0] + self.error_matrix[1][1]) / elements_num
            elif self.metric == "precision":
                ans = self.error_matrix[0][0] / (self.error_matrix[0][0] + self.error_matrix[1][0])
            elif self.metric == "recall":
                ans = self.error_matrix[0][0] / (self.error_matrix[0][0] + self.error_matrix[0][1])
            elif self.metric == "f1":
                ans = 2*self.error_matrix[0][0] / (self.error_matrix[0][0]*2 + self.error_matrix[1][0]+self.error_matrix[0][1])
            elif self.metric == "roc_auc":
                ans = self.integral(self.fpr, self.tpr)
        except ZeroDivisionError:
            ans = 0
        return int(ans * 10000) / 10000


def train_test_split(df, x_cols, y_col, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    df_copy = df.sample(frac=1).reset_index(drop=True)
    train_size = int(np.ceil(df_copy.shape[0] * (1 - test_size)))
    return (df_copy[x_cols][:train_size], df_copy[x_cols][train_size:], df_copy[y_col][:train_size],
            df_copy[y_col][train_size:])


def print_error_matrix_pretty(matrix):
    df_error = pd.DataFrame(matrix, index=["True e", "True p"],
                            columns=["Predicted e", "Predicted p"])
    print(df_error)
    print()


df = pd.read_csv("diabetes.csv")
featuresSet = list(df.drop("Outcome", axis=1).keys())
x_train, x_test, y_train, y_test = train_test_split(df, featuresSet, "Outcome")
DEBUG = False


def test_case(iter_count=10, learning_rate=0.1, learning_rate_func=None, learning_rate_func_str=None):
    logReg = MyLogReg(iter_count, learning_rate=learning_rate, learning_rate_func=learning_rate_func)
    verbose = None
    if DEBUG:
        verbose = iter_count // 100
    logReg.fit(x_train, y_train, verbose=verbose)
    y_pred = logReg.predict(x_test, add_dummy_col=True)
    logReg.error_matrix = logReg.get_error_matrix(y_test, y_pred)
    logReg.generate_roc_data(y_test, logReg.predict_proba(x_test, add_dummy_col=True))
    roc_auc = logReg.integral(logReg.fpr, logReg.tpr)
    print(f"Test case input: iter_count: {iter_count}, ", end="")
    if learning_rate_func:
        print(f"learning_rate_function: {learning_rate_func_str}")
    else:
        print(f"learning_rate: {learning_rate}")
    print()
    logReg.metric = "f1"
    print("f1 score:", logReg.get_best_score())
    logReg.metric = "roc_auc"
    print("roc auc score:", logReg.get_best_score())
    logReg.metric = "accuracy"
    print("accuracy score:", logReg.get_best_score())
    logReg.metric = "recall"
    print("recall score:", logReg.get_best_score())
    logReg.metric = "precision"
    print("precision score:", logReg.get_best_score())
    print_error_matrix_pretty(logReg.error_matrix)
    plt.plot(logReg.fpr, logReg.tpr, label='ROC кривая (area = %0.3f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()


test_case()
test_case(100, 0.001)
test_case(1000, learning_rate_func=lambda iter: 0.5*0.99**iter, learning_rate_func_str="0.5*0.99**iter")
test_case(2000, learning_rate_func=lambda iter: 1e-3 + iter / 2000 * (1e-4 - 1e-3),
          learning_rate_func_str="0.001 + iter/2000 * (0.0001 - 0.001)")
test_case(10000, learning_rate=4e-4)  # roc_auc = 0.68
