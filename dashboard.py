import pandas as pd
from pandas import MultiIndex, Int64Index
from xgboost import XGBClassifier
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
import pickle

pickle_model = pickle.load(open('Credit_model.pkl', 'rb'))



#st.write("""# Application simple pour la prediction
# Cette application est un Dashboard pour le chargé d’étude client
#""")

#Collecter le profil d'entrée
#app_choice = st.sidebar.selectbox("API qu'on souhaite utiliser",["Heroku"])

donnee_entree =pd.read_csv('data_preprocessing2_test.csv')


donnee_entree['CODE_GENDER'] = donnee_entree['CODE_GENDER'].apply(lambda row : 1 if row == "F" else 0)
donnee_entree['FLAG_OWN_CAR'] = donnee_entree['FLAG_OWN_CAR'].apply(lambda row : 1 if row == "Y" else 0)
donnee_entree['FLAG_OWN_REALTY'] = donnee_entree['FLAG_OWN_REALTY'].apply(lambda row : 1 if row == "Y" else 0)
donnee_entree['NAME_CONTRACT_TYPE'] = donnee_entree['NAME_CONTRACT_TYPE'].apply(lambda row : 1 if row == "Cash loans" else 0)
donnee_entree['FLAG_CG_ratio'] = donnee_entree['FLAG_CG_ratio'].apply(lambda row : 1 if row == "True" else 0)
donnee_entree['DAYS_ID_4200'] = donnee_entree['DAYS_ID_4200'].apply(lambda row : 1 if row == "True" else 0)

# encodage des données
var_cat=['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
for col in var_cat:
    dummy=pd.get_dummies(donnee_entree[col],drop_first=True)
    donnee_entree=pd.concat([dummy,donnee_entree],axis=1)
    del donnee_entree[col]

# prendre uniquement la premiere ligne
st.cache(donnee_entree)

def predict(ID, donnee_entree):
    scaler = StandardScaler()
    ID = int(ID)
    Y = donnee_entree[donnee_entree['SK_ID_CURR'] == ID]
    Y = Y.drop(['SK_ID_CURR'], axis=1)
    num = np.array(scaler.fit_transform(Y))
    proba = pickle_model.predict_proba(num)[:,1]
    if proba > 0.1:
        prediction = 'Rejet demande de crédit'
        st.balloons()

    else:
        prediction = 'Acceptation demande de crédit'
        st.balloons()


    return prediction, (proba[0]*100).round()

def main():

    st.sidebar.header("Les caracteristiques du client")

    # front end elements of the web page
    html_temp = """ 
           <div style ="background-color:darkgreen;padding:13px"> 
           <h1 style ="color:black;text-align:center;">L'application qui prédit l'accord du crédit
           </h1> 
           </div> 
           <p></p>
           <div class="txt">Cette application est un Dashboard pour le chargé d’étude client.</div>
           <p></p>
           """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    #col1, col2, col3, col4 = st.columns(4)
    CODE_GENDER = st.sidebar.selectbox("Enter your gender",["M", "F"])
    FLAG_OWN_CAR = st.sidebar.selectbox("Are you have a car",["Y", "N"])
    FLAG_OWN_REALTY = st.sidebar.selectbox("Are you the owner",["Y", "N"])
    AMT_ANNUITY = st.sidebar.number_input("Enter credit by year")

    AMT_GOODS_PRICE = st.sidebar.number_input("Enter your goods price")
    REGION_POPULATION_RELATIVE = st.sidebar.number_input("Enter a goods price")
    FLAG_PHONE = st.sidebar.number_input("Enter flag phone")
    HOUR_APPR_PROCESS_START = st.sidebar.number_input("Enter hour process start")

    EXT_SOURCE_1 = st.sidebar.number_input("Enter your credit history 1")
    EXT_SOURCE_2 = st.sidebar.number_input("Enter your credit history 2")
    EXT_SOURCE_3 = st.sidebar.number_input("Enter your credit history 3")
    FLAG_DOCUMENT_3 = st.sidebar.number_input("Enter document 3")

    PHONE_CHANGE_YEARS = st.sidebar.number_input("Enter phone change years")
    YEARS_EMPLOYED = st.sidebar.number_input("Enter years employed")
    CA_RATIO = st.sidebar.number_input("percentage credit/annuity")
    CG_RATIO = st.sidebar.number_input("credit/per property")

    log_GOODS = st.sidebar.number_input("Enter the amount of goods")
    log_INCOME = st.sidebar.number_input("Enter the amount of income")
    DAYS_ID_4200 = st.sidebar.selectbox("Are you the days registre",["True", "False"])
    NAME_EDUCATION_TYPE = st.sidebar.selectbox("Enter your level education",["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"])


    NAME_CONTRACT_TYPE= st.sidebar.selectbox("Name type contract",['Cash loans','Revolving loans'])
    NAME_TYPE_SUITE = st.sidebar.selectbox("Name type suite",['Unaccompanied','Family','Spouse, partner','Children','Other_B','Other_A','Group of people'])
    NAME_INCOME_TYPE= st.sidebar.selectbox("Name type income",['Working','Commercial associate','Pensioner','State servant','Unemployed','Student','Businessman'])
    NAME_FAMILY_STATUS = st.sidebar.selectbox("Name family status",['Married','Single / not married','Civil marriage','Separated','Widow'])

    AMT_CREDIT = st.sidebar.number_input("Enter your credit")
    DAYS_REGISTRATION = st.sidebar.number_input("Enter days registration")
    DAYS_ID_PUBLISH= st.sidebar.number_input("Enter ID publish")
    OWN_CAR_AGE= st.sidebar.number_input("Enter own car age")

    NAME_HOUSING_TYPE= st.sidebar.selectbox("Name housing type",['House / apartment','With parents','Municipal apartment','Rented apartment','Office apartment','Co-op apartment'])
    OCCUPATION_TYPE = st.sidebar.selectbox("Occupation type",['Laborers','Sales staff','Core staff','Managers','Drivers','High skill tech staff','Accountants','Medicine staff','Security staff','Cooking staff','Cleaning staff','Private service staff','Low-skill Laborers','Waiters/barmen staff','Secretaries','Realty agents','HR staff','IT staff'])
    WEEKDAY_APPR_PROCESS_START= st.sidebar.selectbox("Name type income",['TUESDAY','WEDNESDAY','MONDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY'])
    AMT_INCOME_TOTAL = st.sidebar.number_input("Enter income total")

    FLAG_EMP_PHONE = st.sidebar.number_input("Enter emp phone")
    FLAG_WORK_PHONE = st.sidebar.number_input("Enter work phone")
    REGION_RATING_CLIENT= st.sidebar.number_input("Enter rating client")
    REGION_RATING_CLIENT_W_CITY= st.sidebar.number_input("Enter rating client city")

    REG_CITY_NOT_LIVE_CITY = st.sidebar.number_input("Reg city not live city")
    REG_CITY_NOT_WORK_CITY = st.sidebar.number_input("Reg city not work city")
    LIVE_CITY_NOT_WORK_CITY= st.sidebar.number_input("Live city not work city")
    YEARS_BEGINEXPLUATATION_AVG= st.sidebar.number_input("Years begin expluatation")

    FLAG_DOCUMENT_6	= st.sidebar.number_input("Enter document 6")
    AGE = st.sidebar.number_input("Enter your age")
    EA_RATIO= st.sidebar.number_input("Time worked/age")
    CI_RATIO= st.sidebar.number_input("credit/income percentage")

    AI_RATIO = st.sidebar.number_input("annuity/income percentage")
    log_ANNUITY	= st.sidebar.number_input("average annual income")
    log_CREDIT= st.sidebar.number_input("Enter average credit")
    FLAG_CG_ratio  = st.sidebar.number_input("ratio credit/per property")
    resultat = ""

    st.subheader('Résultat de la prévision')
    if st.button("Predict"):
        resultat = predict(100038, donnee_entree)
        st.success('Votre prêt est {}'.format(resultat))
        #st.balloons()


if __name__ == '__main__':
    main()

#donnee_entree = donnee_entree.drop(['SK_ID_CURR'], axis=1)
#col = st.multiselect("Select a Column", donnee_entree.columns)

#plt.plot(donnee_entree['EXT_SOURCE_1'], donnee_entree[col])

#st.pyplot()















