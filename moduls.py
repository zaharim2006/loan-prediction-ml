from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import data_handler as dh


def models():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')

    model1 = SVC(kernel='rbf')
    model1.fit(X_train_scaled, y_train)

    model2 = LogisticRegression()
    model2.fit(X_train_scaled, y_train)

    model3 = RandomForestClassifier(n_estimators=100, max_depth=4)
    model3.fit(X_train_scaled, y_train)

    model4 = KNeighborsClassifier(n_neighbors=5)
    model4.fit(X_train_scaled, y_train)

    model5 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=1, random_state=0)
    model5.fit(X_train_scaled, y_train)

    model6 = XGBClassifier(eval_metric='logloss')
    model6.fit(X_train_scaled, y_train)

    model7 = DecisionTreeClassifier()
    model7.fit(X_train_scaled, y_train)

    return model1, model2, model3, model4, model5, model6, model7