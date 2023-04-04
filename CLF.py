import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from segdataset_trad import traditional_seg,traditional_tree
from sklearn import tree
import graphviz
train_x,train_y=traditional_tree(type='train')
test_x,test_y=traditional_tree(type='test')

# model = SVC(C=20.0, gamma=0.00001)#0.8
# model = SVC(C=1, gamma=0.00001)#0.675
# model = SVC(C=5, gamma=0.00001)#0.7875

# model = SVC(C=1)#0.825
def tree3(clf):
    fig = plt.figure(figsize=(35, 10))
    tree.plot_tree(clf, fontsize=16)
    fig.savefig("tree.png")

model = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)

# 用训练集训练：
model.fit(train_x, train_y)
# dot_data = tree.export_graphviz(model, out_file=None,
#                      feature_names=['mfccs',' chroma', 'mel', 'contrast', 'var'],
#                      # class_names=iris.target_names,
#                       filled=True, rounded=True,
#                       special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("iris")

tree3(model)
# 用测试集预测：
prediction = model.predict(test_x)
print('准确率：', metrics.accuracy_score(prediction, test_y))
confusion=confusion_matrix(test_y,prediction)
print(confusion)
