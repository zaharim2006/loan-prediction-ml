import streamlit as st
import data_handler as dh
import pandas as pd
import numpy as np

from train import Models
import moduls as mls


st.markdown("<h1 style='text-align: center; color: coral;'>Kredit Tasdiqlash Bashoratchisi!</h1>", unsafe_allow_html=True)

df = pd.read_csv('train.csv')

data_ = st.sidebar.radio(
    label="Barcha ma'lumotlarni ko'rmoqchimisiz?",
    options=("Yo'q", "Ha"))

if data_ == 'Ha':
    st.markdown("<h1 style='text-align: center; color: red;'>Dataset</h1>", unsafe_allow_html=True)
    st.write(df)

st.sidebar.subheader("Algoritmlar")
top_book_ = st.sidebar.selectbox(
    label="Algoritmni tanlang",
    options=['SVC', 'LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier',
             'GradientBoostingClassifier', 'XGBClassifier', 'DecisionTreeClassifier'])

models_acc = [Models.svc(), Models.lr(), Models.rfc(), Models.knc(), Models.gbc(), Models.xgbc(), Models.dtc()]

a = ['SVC', 'LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier',
     'GradientBoostingClassifier', 'XGBClassifier', 'DecisionTreeClassifier']

if top_book_:
    st.text("\n\n")
    st.write(models_acc[a.index(top_book_)])

try:
    user_input = st.sidebar.radio(
        label="O'z kreditingiz tasdiqlanishini bilmoqchimisiz?",
        options=("Yo'q", "Ha"))

    if user_input == 'Ha':

        st.title('Kredit Bashoratchisi')
        text = """
            <div style='background-color:tomato;padding:10px'>
            <h4 style='color:white;text-align:center;'>Streamlit Loan Prediction ML App</h4>
            </div>
        """
        st.markdown(text, unsafe_allow_html=True)
        st.title('')
        st.markdown('<h5>Iltimos, quyidagi savollarga javob bering!</h5>', unsafe_allow_html=True)
        st.title('')

        gender_ = st.radio(label="Jinsingiz", options=('Erkak', 'Ayol'))
        gender = 1 if gender_ == 'Erkak' else 0

        married_ = st.radio(label="Turmush holatingiz", options=('Ha', "Yo'q"))
        married = 1 if married_ == 'Ha' else 0

        dependents_ = st.radio(label="Qaramog'ingizdagilar soni", options=('0', '1', '2', '3+'))
        dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
        dependents = dep_map[dependents_]

        education_ = st.radio(label="Ta'lim darajangiz", options=("Oliy ma'lumotli", "Oliy ma'lumotsiz"))
        education = 0 if education_ == "Oliy ma'lumotli" else 1

        self_employed_ = st.radio(label="O'zingizga ish beruvchimisiz?", options=("Yo'q", 'Ha'))
        self_employed = 1 if self_employed_ == 'Ha' else 0

        applicant_income = st.text_input("Oylik daromadingiz", value="0")

        coapplicant_income = st.text_input("Turmush o'rtog'ingizning daromadi (bo'lmasa 0)", value="0")

        loan_amount = st.text_input("So'ralayotgan kredit miqdori (minglarda)", value="0")

        loan_term = st.text_input("Kredit muddati (kunlarda, masalan 360)", value="360")

        credit_history = st.radio(label="Kredit tarixingiz yaxshimi?", options=(1.0, 0.0))

        property_area_ = st.radio(label="Yashash hududingiz", options=('Shahar', 'Yarim shahar', 'Qishloq'))
        area_map = {'Qishloq': 0, 'Yarim shahar': 1, 'Shahar': 2}
        property_area = area_map[property_area_]

        x = pd.DataFrame({
            "Gender": [gender], "Married": [married], "Dependents": [dependents], "Education": [education],
            "Self_Employed": [self_employed], "ApplicantIncome": [float(applicant_income)],
            "CoapplicantIncome": [float(coapplicant_income)], "LoanAmount": [float(loan_amount)],
            "Loan_Amount_Term": [float(loan_term)], "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess('./train.csv')
        x_scaled = scaler.transform(x)
        predictions = np.array([model.predict(x_scaled) for model in mls.models()])

        predict = st.button("Bashorat qilish")
        if predict:
            if predictions.mean() < 0.5:
                st.markdown("<h3 style='text-align: center; color: blue;'>Afsuski, kredit rad etilishi mumkin</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: blue;'>Tabriklaymiz! Kredit tasdiqlanishi mumkin</h3>", unsafe_allow_html=True)

except ValueError:
    st.error("Iltimos, to'g'ri ma'lumot kiriting!")

about = st.sidebar.button("Muallif")
if about:
    st.title("Zaharim")
    st.markdown("<h1 style='text-align: center; color: blue;'>E'tiboringiz uchun rahmat!</h1>", unsafe_allow_html=True)