from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

import data_handler as dh
from sklearn.metrics import accuracy_score as score


class Models:

    def svc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = SVC(kernel='rbf')
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f'SVM.SVC aniqligi:      {acc * 100:.2f} %'

    def lr():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f'Logistik regressiya aniqligi:       {acc * 100:.2f} %'

    def rfc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = RandomForestClassifier(n_estimators=100, max_depth=4)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f"Tasodifiy o'rmon aniqligi:        {acc * 100:.2f} %"

    def knc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f"Eng yaqin qo'shnilar (KNN) aniqligi:       {acc * 100:.2f} %"

    def gbc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=1, random_state=0)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f'Gradient kuchaytirish aniqligi:      {acc * 100:.2f} %'

    def xgbc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f'XGBoost aniqligi:       {acc * 100:.2f} %'

    def dtc():
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        model = DecisionTreeClassifier()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = score(y_test, preds)
        return f'Qaror daraxti aniqligi:        {acc * 100:.2f} %'