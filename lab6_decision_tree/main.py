import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
import random
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt


# Just stub for typing
class TreeNode:
    pass


class EdgeWithCondition:

    def __init__(self, feature: str, value: str):
        self.feature = feature
        self.value = value

    def checkElement(self, element):
        return element[self.feature] == self.value

    def __str__(self):
        return str(self.__dict__)


class TreeNode:

    def __init__(self):
        self.poisonous = None
        self.edgesToChildren: Dict[EdgeWithCondition, TreeNode] = {}
        self.probability = 0

    def addEdge(self, edge: EdgeWithCondition, child: TreeNode):
        self.edgesToChildren[edge] = child

    def getNextNode(self, element: pd.Series):
        for edge, child in self.edgesToChildren.items():
            if edge.checkElement(element):
                return child
        print({i.__str__(): j for i, j in self.edgesToChildren.items()})
        raise RuntimeError(f"Can not find path for element {element}")

    def __str__(self):
        d = self.__dict__.copy()
        d["edgesToChildren"] = {i.__str__(): j.__str__() for i, j in self.edgesToChildren.items()}
        return str(d)


def getInfo(data: pd.DataFrame):
    global target

    ans = 0
    for target_value in set(data[target].values.tolist()):
        ratio = len(data.loc[data[target] == target_value]) / len(data)
        ans += ratio * np.log2(ratio)
    return -ans


def getBestFeature(data: pd.DataFrame):
    max_gain_ratio = -1e9
    max_gain_ratio_feature = None
    for feature in data.keys():
        cur_entropy = getInfo(data)
        cur_conditional_entropy = 0
        cur_split = 0
        if feature == target:
            continue
        for value in set(data[feature].values.tolist()):
            cur_elements = data.loc[data[feature] == value]
            cur_coef = len(cur_elements) / len(data)
            cur_split += cur_coef * np.log2(cur_coef)
            cur_conditional_entropy += getInfo(cur_elements) * cur_coef
        cur_gain_ratio = (cur_entropy - cur_conditional_entropy)
        if cur_gain_ratio > max_gain_ratio:
            max_gain_ratio = cur_gain_ratio
            max_gain_ratio_feature = feature
    return max_gain_ratio_feature


def buildTree(data: pd.DataFrame, cur_node: TreeNode, depth=0):
    global target

    # Only one class left
    if len(set(data[target].values)) == 1:
        cur_node.poisonous = data[target].values[0]
        cur_node.probability = 1
        return

    # Since we are reducing number of features at the beginning there maybe a situation when all the features in
    # all the rows are the same but target fields are not
    if len(set("$".join(row.values.tolist()[:-1]) for _, row in data.iterrows())) == 1:
        target_value_with_max_count = None
        target_value_max_count = 0
        for target_value in set(data[target].values.tolist()):
            cur_count = len(data.loc[data[target] == target_value])
            if cur_count > target_value_max_count:
                target_value_max_count = cur_count
                target_value_with_max_count = target_value
        cur_node.poisonous = target_value_with_max_count
        # For roc-auc
        cur_node.probability = target_value_max_count / len(data)
        return

    best_feature = getBestFeature(data)
    for value in set(data[best_feature].values.tolist()):
        edge = EdgeWithCondition(best_feature, value)
        child_node = TreeNode()
        cur_node.addEdge(edge, child_node)
        # maybe add .copy
        buildTree(data.loc[data[best_feature] == value], child_node, depth + 1)


def classify(element: pd.Series, head: TreeNode):
    while True:
        if len(head.edgesToChildren.keys()) == 0:
            return head.poisonous, head.probability
        head = head.getNextNode(element)


def train_test_split(df, x_cols, y_col, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    df_copy = df.sample(frac=1).reset_index(drop=True)
    train_size = int(np.ceil(df_copy.shape[0] * (1 - test_size)))
    return (df_copy[x_cols][:train_size], df_copy[x_cols][train_size:], df_copy[y_col][:train_size],
            df_copy[y_col][train_size:])


def classifyAll(x_test: pd.DataFrame, head: TreeNode) -> List[Tuple[str, float]]:
    return [classify(row, head) for _, row in x_test.iterrows()]


def get_error_matrix(y_true, y_pred):
    classes = ["e", "p"]
    matrix = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
    for i, actual_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i][j] = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t == actual_class and y_p == pred_class)
    return matrix


def get_fpr_and_tpr(y_true, y_pred):
    matrix = get_error_matrix(y_true, y_pred)
    tp = matrix[0][0]
    fp = matrix[1][0]
    fn = matrix[0][1]
    tn = matrix[1][1]
    return fp / (fp + tn), tp / (tp + fn)


def get_y_pred_from_probs(y_true_probs, threshold):
    return ["e" if i >= threshold else "p" for i in y_true_probs]


def print_error_matrix_pretty(matrix):
    df_error = pd.DataFrame(matrix, index=["True e", "True p"],
                            columns=["Predicted e", "Predicted p"])
    print(df_error)
    print()
    number_of_elements = sum(matrix[0]) + sum(matrix[1])
    accuracy = round((matrix[0][0] + matrix[1][1]) / number_of_elements, 3)
    precision = round(matrix[0][0] / (matrix[0][0] + matrix[1][0]), 3)
    recall = round(matrix[0][0] / sum(matrix[0]), 3)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}")


def roc_curve(y_test: List[str], y_probs: List[float], points_num=1000) -> Tuple[List[float], List[float]]:
    fpr = []
    tpr = []
    for i in range(points_num):
        cur_threshold = 1 - i / points_num
        cur_fpr, cur_tpr = get_fpr_and_tpr(y_test, get_y_pred_from_probs(y_probs, cur_threshold))
        fpr.append(cur_fpr)
        tpr.append(cur_tpr)
    return fpr, tpr


def pr_curve(y_test: List[str], y_probs: List[float], points_num=1000) -> Tuple[List[float], List[float]]:
    precisions = []
    recalls = []
    for i in range(points_num):
        cur_threshold = 1 - i / points_num
        cur_y_pred = get_y_pred_from_probs(y_probs, cur_threshold)
        matrix = get_error_matrix(y_test, cur_y_pred)
        precision = round(matrix[0][0] / (matrix[0][0] + matrix[1][0]), 3)
        recall = round(matrix[0][0] / sum(matrix[0]), 3)
        precisions.append(precision)
        recalls.append(recall)
    return recalls, precisions


def integral(xs: List[float], ys: List[float]) -> float:
    # Используем метод средних прямоугольников
    ans = 0
    for i in range(len(ys) - 1):
        ans += (xs[i + 1] - xs[i]) * (ys[i+1] + ys[i]) / 2
    return ans


mushroom = fetch_ucirepo(id=73)
X: pd.DataFrame = mushroom.data.features
full: pd.DataFrame = mushroom.data.original
target = "poisonous"

# Only column with nan
full.loc[full['stalk-root'].isnull(), 'stalk-root'] = ''

# Get sqrt(n) random features
featuresSet = list(X.keys())
random.shuffle(featuresSet)
featuresSet = featuresSet[:int(len(featuresSet)**.5)]
print(f"Current reduced feature set: {featuresSet}")

x_train, x_test, y_train, y_test = train_test_split(full, featuresSet, [target])

HEAD = TreeNode()
buildTree(pd.concat([x_train, y_train], axis=1), HEAD)
y_classified_with_probs = classifyAll(x_test, HEAD)
y_classified = [i[0] for i in y_classified_with_probs]
y_true_probs = [i[1] if i[0] == "e" else 1 - i[1] for i in y_classified_with_probs]
y_test = [i[0] for i in y_test.values]
error_matrix = get_error_matrix(y_test, y_classified)
print_error_matrix_pretty(error_matrix)
fpr, tpr = roc_curve(y_test, y_true_probs)
fpr.append(1)
tpr.append(tpr[-1])
roc_auc = integral(fpr, tpr)
plt.plot(fpr, tpr, label='ROC кривая (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

r, p = pr_curve(y_test, y_true_probs)
p.insert(0, 1)
r.insert(0, 0)
pr_auc = integral(r, p)
plt.plot(r, p, label='Precision-recall кривая (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-recall кривая')
plt.legend(loc="lower right")
plt.show()

