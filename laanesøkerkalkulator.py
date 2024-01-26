# Importering av pakker
import streamlit as st
import pandas as pd
import shap
import joblib
from dataanalyse_og_preprosessering import data_analysis


# Importér den trente modellen
loaded_model = joblib.load('trained_model.joblib')

# Henter variabler fra funksjonen data_analysis() i filen dataanalyse_og_preprosessering.py
train, test, train_encoded, test_encoded, x_train, y_train, X_train, X_val, y_val = data_analysis()

# Streamlit
st.sidebar.header("Input-verdier")
def verdier_fra_bruker(train):
    data = {}
    # "Felter"-dictionarien inneholder feltnavnene som nøkler og deres respektive
    # kolonne som verdier. I tilfeller der et felt refererer til flere kolonner,
    # er verdien en dictionary som inneholder feltalternativenes navn som nøkler og de respektive kolonnenavnene som verdier.
    felter = {"Applicant Income": "ApplicantIncome",
              "Coapplicant Income": "CoapplicantIncome",
              "Loan Amount": "LoanAmount",
              "Loan Amount Term": "Loan_Amount_Term",
              "Credit History": "Credit_History",
              "Gender": "Gender_Male",
              "Married": "Married_Yes",
              "Education": "Education_Not Graduate",
              "Employment": "Self_Employed_Yes",
              "Dependents": {"0": "", "1": "Dependents_1", "2": "Dependents_2", "3+": "Dependents_3+"},
              "Property Area": {"Rural": "", "Semiurban": "Property_Area_Semiurban", "Urban": "Property_Area_Urban"}}
    
    # Dictionarien "felter_options" holder informasjonen om feltene som har 
    # flere alternativer, men som kun refererer til en enkelt kolonne.
    felter_options = {"Gender": {"Female": 0, "Male": 1},
                "Married": {"Unmarried": 0, "Married": 1},
                "Education": {"Not graduate": 0, "Graduate": 1},
                "Credit History": {"Has no history": 0, "Has history": 1},
                "Employment": {"Not self employed": 0, "Self employed": 1}}
    
    for felt in felter.keys():
        kolonne_navn = felter[felt]
        if type(kolonne_navn) is dict:
            selected_label = st.sidebar.selectbox(felt, kolonne_navn.keys())
            value = kolonne_navn[selected_label]
            
            feltets_kolonnenavn = [value for key, value in kolonne_navn.items() if key != ""]
            for option in feltets_kolonnenavn:
                if option != "":
                    if value == option:
                        data[option] = 1
                    else:
                        data[option] = 0
        else:
            if train[kolonne_navn].dtype == "uint8":
                feltets_options = felter_options[felt]
                selected_label = st.sidebar.selectbox(felt, list(feltets_options.keys()))
                value = feltets_options[selected_label]
            else:
                min_value, max_value, default_value = float(train[kolonne_navn].min()), float(train[kolonne_navn].max()), float(train[kolonne_navn].mean())
                value = st.sidebar.slider(felt, min_value, max_value, default_value)
            data[kolonne_navn] = value
        
    data_df = pd.DataFrame(columns=train.columns)
    new_row = pd.DataFrame([data])
    data_df = pd.concat([data_df, new_row], ignore_index=True)
    return data_df


input_data = verdier_fra_bruker(test_encoded)


# Bruk SHAP for å forklare modellens utdata for den gitte modell og inputverdier.
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(input_data)

# Undertrykk den kommende advarselen
st.set_option('deprecation.showPyplotGlobalUse', False)
# Plot force plot
force_plot = shap.plots.force(explainer.expected_value[1], shap_values[0], feature_names=input_data.columns, matplotlib=True)

st.header("Automatisk vurderingssystem for lånesøknader")

st.write("""
         Utviklet av BASTIAN UNDHEIM ØIAN, HARALD NORVALD STABBETORP OG SINDRE STOKKE""")

st.write("Input-verdier")
st.table(input_data)
st.write("---")

user_prediction = loaded_model.predict(input_data)
if (user_prediction == 0):
    st.write("Basert på dine tall, vil lånet ditt bli godkjent")
else:
    st.write("Basert på dine tall, vil lånet ditt ikke bli godkjent")

st.pyplot(force_plot)

