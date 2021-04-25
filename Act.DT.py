import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(pd.read_csv("SkyServer.csv"), test_size=0.2, random_state=52)
data = train_data
print(data.head())

features = data.drop(['class'], axis=1)
features = pd.get_dummies(features, drop_first=True)
print(features)

target = data['class'].values
target = pd.get_dummies(target, drop_first=True)

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
trained_tree = decision_tree.fit(features, target)

tree.plot_tree(trained_tree, feature_names=features.columns)
plt.show()
