from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

iris = load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Doğruluk Skoru:", accuracy_score(y_test,y_pred))
print("\nsınıflandırma Raporu:\n",classification_report(y_test,y_pred,target_names=iris.target_names))

plt.figure(figsize=(20,10))
plot_tree(model,
          filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True,
          fontsize=24)
plt.title("Karar Agac Gorsellestirmesi - Iris Veri Seti")
plt.show()

print(iris.data[:5])
