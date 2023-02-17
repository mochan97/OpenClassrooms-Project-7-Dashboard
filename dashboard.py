import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import urllib.request
#import os
#import ssl
import ast

#FastAPI_URL = 'http://127.0.0.1:8000/'#local URL
#FastAPI_URL = 'https://openclassrooms-project7-api.azurewebsites.net/'#Azure URL

df = pd.read_csv('df_sample_for_dashboard.csv')
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)

def prediction(input_data):
    data =  {
    "input_data": {
        "columns": [
            "AMT_ANNUITY",
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "AMT_INCOME_TOTAL",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "ANNUITY_INCOME_PERC",
            "APPROVED_AMT_ANNUITY_MAX",
            "APPROVED_AMT_ANNUITY_MEAN",
            "APPROVED_AMT_ANNUITY_MIN",
            "APPROVED_AMT_APPLICATION_MAX",
            "APPROVED_AMT_APPLICATION_MEAN",
            "APPROVED_AMT_APPLICATION_MIN",
            "APPROVED_AMT_CREDIT_MAX",
            "APPROVED_AMT_CREDIT_MEAN",
            "APPROVED_AMT_CREDIT_MIN",
            "APPROVED_AMT_DOWN_PAYMENT_MAX",
            "APPROVED_AMT_DOWN_PAYMENT_MEAN",
            "APPROVED_AMT_DOWN_PAYMENT_MIN",
            "APPROVED_AMT_GOODS_PRICE_MAX",
            "APPROVED_AMT_GOODS_PRICE_MEAN",
            "APPROVED_AMT_GOODS_PRICE_MIN",
            "APPROVED_APP_CREDIT_PERC_MAX",
            "APPROVED_APP_CREDIT_PERC_MEAN",
            "APPROVED_APP_CREDIT_PERC_MIN",
            "APPROVED_CNT_PAYMENT_MEAN",
            "APPROVED_CNT_PAYMENT_SUM",
            "APPROVED_DAYS_DECISION_MAX",
            "APPROVED_DAYS_DECISION_MEAN",
            "APPROVED_DAYS_DECISION_MIN",
            "APPROVED_HOUR_APPR_PROCESS_START_MAX",
            "APPROVED_HOUR_APPR_PROCESS_START_MEAN",
            "APPROVED_HOUR_APPR_PROCESS_START_MIN",
            "APPROVED_RATE_DOWN_PAYMENT_MAX",
            "APPROVED_RATE_DOWN_PAYMENT_MEAN",
            "APPROVED_RATE_DOWN_PAYMENT_MIN",
            "BURO_AMT_CREDIT_SUM_DEBT_MAX",
            "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
            "BURO_AMT_CREDIT_SUM_DEBT_SUM",
            "BURO_AMT_CREDIT_SUM_LIMIT_SUM",
            "BURO_AMT_CREDIT_SUM_MAX",
            "BURO_AMT_CREDIT_SUM_MEAN",
            "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN",
            "BURO_AMT_CREDIT_SUM_SUM",
            "BURO_CNT_CREDIT_PROLONG_SUM",
            "BURO_CREDIT_DAY_OVERDUE_MAX",
            "BURO_CREDIT_DAY_OVERDUE_MEAN",
            "BURO_DAYS_CREDIT_ENDDATE_MAX",
            "BURO_DAYS_CREDIT_ENDDATE_MEAN",
            "BURO_DAYS_CREDIT_ENDDATE_MIN",
            "BURO_DAYS_CREDIT_MAX",
            "BURO_DAYS_CREDIT_MEAN",
            "BURO_DAYS_CREDIT_MIN",
            "BURO_DAYS_CREDIT_UPDATE_MEAN",
            "BURO_MONTHS_BALANCE_SIZE_SUM",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "CODE_GENDER",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_EMPLOYED_PERC",
            "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE",
            "DAYS_REGISTRATION",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "EMERGENCYSTATE_MODE",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "FLAG_CONT_MOBILE",
            "FLAG_DOCUMENT_10",
            "FLAG_DOCUMENT_11",
            "FLAG_DOCUMENT_12",
            "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14",
            "FLAG_DOCUMENT_15",
            "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17",
            "FLAG_DOCUMENT_18",
            "FLAG_DOCUMENT_19",
            "FLAG_DOCUMENT_2",
            "FLAG_DOCUMENT_20",
            "FLAG_DOCUMENT_21",
            "FLAG_DOCUMENT_3",
            "FLAG_DOCUMENT_4",
            "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_7",
            "FLAG_DOCUMENT_8",
            "FLAG_DOCUMENT_9",
            "FLAG_EMAIL",
            "FLAG_EMP_PHONE",
            "FLAG_MOBIL",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "FLAG_PHONE",
            "FLAG_WORK_PHONE",
            "FONDKAPREMONT_MODE",
            "HOUR_APPR_PROCESS_START",
            "HOUSETYPE_MODE",
            "INCOME_CREDIT_PERC",
            "INCOME_PER_PERSON",
            "INSTAL_AMT_INSTALMENT_MAX",
            "INSTAL_AMT_INSTALMENT_MEAN",
            "INSTAL_AMT_INSTALMENT_SUM",
            "INSTAL_AMT_PAYMENT_MAX",
            "INSTAL_AMT_PAYMENT_MEAN",
            "INSTAL_AMT_PAYMENT_MIN",
            "INSTAL_AMT_PAYMENT_SUM",
            "INSTAL_COUNT",
            "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
            "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
            "INSTAL_DAYS_ENTRY_PAYMENT_SUM",
            "INSTAL_DBD_MAX",
            "INSTAL_DBD_MEAN",
            "INSTAL_DBD_SUM",
            "INSTAL_DPD_MAX",
            "INSTAL_DPD_MEAN",
            "INSTAL_DPD_SUM",
            "INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE",
            "INSTAL_PAYMENT_DIFF_MAX",
            "INSTAL_PAYMENT_DIFF_MEAN",
            "INSTAL_PAYMENT_DIFF_SUM",
            "INSTAL_PAYMENT_DIFF_VAR",
            "INSTAL_PAYMENT_PERC_MAX",
            "INSTAL_PAYMENT_PERC_MEAN",
            "INSTAL_PAYMENT_PERC_SUM",
            "INSTAL_PAYMENT_PERC_VAR",
            "LIVE_CITY_NOT_WORK_CITY",
            "LIVE_REGION_NOT_WORK_REGION",
            "NAME_CONTRACT_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_TYPE_SUITE",
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "PAYMENT_RATE",
            "POS_COUNT",
            "POS_MONTHS_BALANCE_MAX",
            "POS_MONTHS_BALANCE_MEAN",
            "POS_MONTHS_BALANCE_SIZE",
            "POS_SK_DPD_DEF_MAX",
            "POS_SK_DPD_DEF_MEAN",
            "POS_SK_DPD_MAX",
            "POS_SK_DPD_MEAN",
            "PREV_AMT_ANNUITY_MAX",
            "PREV_AMT_ANNUITY_MEAN",
            "PREV_AMT_ANNUITY_MIN",
            "PREV_AMT_APPLICATION_MAX",
            "PREV_AMT_APPLICATION_MEAN",
            "PREV_AMT_APPLICATION_MIN",
            "PREV_AMT_CREDIT_MAX",
            "PREV_AMT_CREDIT_MEAN",
            "PREV_AMT_CREDIT_MIN",
            "PREV_AMT_DOWN_PAYMENT_MAX",
            "PREV_AMT_DOWN_PAYMENT_MEAN",
            "PREV_AMT_DOWN_PAYMENT_MIN",
            "PREV_AMT_GOODS_PRICE_MAX",
            "PREV_AMT_GOODS_PRICE_MEAN",
            "PREV_AMT_GOODS_PRICE_MIN",
            "PREV_APP_CREDIT_PERC_MAX",
            "PREV_APP_CREDIT_PERC_MEAN",
            "PREV_APP_CREDIT_PERC_MIN",
            "PREV_CNT_PAYMENT_MEAN",
            "PREV_CNT_PAYMENT_SUM",
            "PREV_DAYS_DECISION_MAX",
            "PREV_DAYS_DECISION_MEAN",
            "PREV_DAYS_DECISION_MIN",
            "PREV_HOUR_APPR_PROCESS_START_MAX",
            "PREV_HOUR_APPR_PROCESS_START_MEAN",
            "PREV_HOUR_APPR_PROCESS_START_MIN",
            "PREV_RATE_DOWN_PAYMENT_MAX",
            "PREV_RATE_DOWN_PAYMENT_MEAN",
            "PREV_RATE_DOWN_PAYMENT_MIN",
            "REGION_POPULATION_RELATIVE",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "SK_ID_CURR",
            "WALLSMATERIAL_MODE",
            "WEEKDAY_APPR_PROCESS_START",
            "index"
            ],
            "index": [0],
            "data": input_data
        }
    }

    body = str.encode(json.dumps(data))

    #url = 'https://openclassrooms-project7-ap-ypbpj.francecentral.inference.ml.azure.com/score'#API Endpoint - Predict Model
    url = 'https://ocr-p7-api-mlflow-proba-dpxsv.francecentral.inference.ml.azure.com/score'#API Endpoint - Predict_Proba Model
    
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    #api_key = 'Ckvgt9gChJs96hla3xYdTXwX3OJ8zgcu' #Key for Predict Model
    api_key = 'vu61TEphLD3l28PHotowYM281tGlWghr' #Key for Predict_proba Model
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'api-mlflow-proba-1' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        
        result = response.read()
        print(result)
    except urllib.error.HTTPError as error:
        
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
    res_decoded = result.decode()
    res_str_list = ast.literal_eval(res_decoded)
    res_proba = res_str_list[0][1]
    return res_proba


# def post_prediction(idClient: int):
#     df_client = df.loc[df['index'] == idClient]
#     dict_client = df_client.to_dict('records')[0]
#     json_client = json.dumps(dict_client)
#     #st.write(json_client)// pour debuggage
#     #URL = FastAPI_URL + 'prediction/'// pour debuggage
#     #st.write(URL) // pour debuggage
#     response = requests.post(FastAPI_URL + 'prediction/', data = json_client)
#     proba = eval(response.content)["probability"]
#     return proba

# def get_threshold():
#     response = requests.get(FastAPI_URL + 'threshold/')
#     return round(float(response.content), 3)

def main():

    st.set_page_config(page_title='Dashboard Application Crédit',
                    layout='centered',
                    initial_sidebar_state='expanded')

    st.title("Application Crédit")

    with st.sidebar:
        idClient = st.selectbox(label = 'Choisir un client', options = df['index'], key='idClient')

    st.write("L'ID du client sélectionné est: ", idClient)


    if st.button('Prédire'):
        #res = post_prediction(idClient)
        #st.write(idClient)
        data_csv = pd.read_csv('df_sample_for_dashboard.csv', index_col=0)
        #st.write(data_csv)
        data_2 = data_csv.copy() 
        data_2.reset_index(drop=True, inplace=True)
        #st.write(data_2)
        data_client = data_2.loc[data_2['index'] == idClient]
        #st.write(data_client)
        data_x = np.asarray(data_client).tolist()
        #st.write(data_x)

        # data_f =  {
        #             "input_data":
        #             {
        #                 "columns": [
        #                     "AMT_ANNUITY",
        #                     "AMT_CREDIT",
        #                     "AMT_GOODS_PRICE",
        #                     "AMT_INCOME_TOTAL",
        #                     "AMT_REQ_CREDIT_BUREAU_DAY",
        #                     "AMT_REQ_CREDIT_BUREAU_HOUR",
        #                     "AMT_REQ_CREDIT_BUREAU_MON",
        #                     "AMT_REQ_CREDIT_BUREAU_QRT",
        #                     "AMT_REQ_CREDIT_BUREAU_WEEK",
        #                     "AMT_REQ_CREDIT_BUREAU_YEAR",
        #                     "ANNUITY_INCOME_PERC",
        #                     "APPROVED_AMT_ANNUITY_MAX",
        #                     "APPROVED_AMT_ANNUITY_MEAN",
        #                     "APPROVED_AMT_ANNUITY_MIN",
        #                     "APPROVED_AMT_APPLICATION_MAX",
        #                     "APPROVED_AMT_APPLICATION_MEAN",
        #                     "APPROVED_AMT_APPLICATION_MIN",
        #                     "APPROVED_AMT_CREDIT_MAX",
        #                     "APPROVED_AMT_CREDIT_MEAN",
        #                     "APPROVED_AMT_CREDIT_MIN",
        #                     "APPROVED_AMT_DOWN_PAYMENT_MAX",
        #                     "APPROVED_AMT_DOWN_PAYMENT_MEAN",
        #                     "APPROVED_AMT_DOWN_PAYMENT_MIN",
        #                     "APPROVED_AMT_GOODS_PRICE_MAX",
        #                     "APPROVED_AMT_GOODS_PRICE_MEAN",
        #                     "APPROVED_AMT_GOODS_PRICE_MIN",
        #                     "APPROVED_APP_CREDIT_PERC_MAX",
        #                     "APPROVED_APP_CREDIT_PERC_MEAN",
        #                     "APPROVED_APP_CREDIT_PERC_MIN",
        #                     "APPROVED_CNT_PAYMENT_MEAN",
        #                     "APPROVED_CNT_PAYMENT_SUM",
        #                     "APPROVED_DAYS_DECISION_MAX",
        #                     "APPROVED_DAYS_DECISION_MEAN",
        #                     "APPROVED_DAYS_DECISION_MIN",
        #                     "APPROVED_HOUR_APPR_PROCESS_START_MAX",
        #                     "APPROVED_HOUR_APPR_PROCESS_START_MEAN",
        #                     "APPROVED_HOUR_APPR_PROCESS_START_MIN",
        #                     "APPROVED_RATE_DOWN_PAYMENT_MAX",
        #                     "APPROVED_RATE_DOWN_PAYMENT_MEAN",
        #                     "APPROVED_RATE_DOWN_PAYMENT_MIN",
        #                     "BURO_AMT_CREDIT_SUM_DEBT_MAX",
        #                     "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
        #                     "BURO_AMT_CREDIT_SUM_DEBT_SUM",
        #                     "BURO_AMT_CREDIT_SUM_LIMIT_SUM",
        #                     "BURO_AMT_CREDIT_SUM_MAX",
        #                     "BURO_AMT_CREDIT_SUM_MEAN",
        #                     "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN",
        #                     "BURO_AMT_CREDIT_SUM_SUM",
        #                     "BURO_CNT_CREDIT_PROLONG_SUM",
        #                     "BURO_CREDIT_DAY_OVERDUE_MAX",
        #                     "BURO_CREDIT_DAY_OVERDUE_MEAN",
        #                     "BURO_DAYS_CREDIT_ENDDATE_MAX",
        #                     "BURO_DAYS_CREDIT_ENDDATE_MEAN",
        #                     "BURO_DAYS_CREDIT_ENDDATE_MIN",
        #                     "BURO_DAYS_CREDIT_MAX",
        #                     "BURO_DAYS_CREDIT_MEAN",
        #                     "BURO_DAYS_CREDIT_MIN",
        #                     "BURO_DAYS_CREDIT_UPDATE_MEAN",
        #                     "BURO_MONTHS_BALANCE_SIZE_SUM",
        #                     "CNT_CHILDREN",
        #                     "CNT_FAM_MEMBERS",
        #                     "CODE_GENDER",
        #                     "DAYS_BIRTH",
        #                     "DAYS_EMPLOYED",
        #                     "DAYS_EMPLOYED_PERC",
        #                     "DAYS_ID_PUBLISH",
        #                     "DAYS_LAST_PHONE_CHANGE",
        #                     "DAYS_REGISTRATION",
        #                     "DEF_30_CNT_SOCIAL_CIRCLE",
        #                     "DEF_60_CNT_SOCIAL_CIRCLE",
        #                     "EMERGENCYSTATE_MODE",
        #                     "EXT_SOURCE_2",
        #                     "EXT_SOURCE_3",
        #                     "FLAG_CONT_MOBILE",
        #                     "FLAG_DOCUMENT_10",
        #                     "FLAG_DOCUMENT_11",
        #                     "FLAG_DOCUMENT_12",
        #                     "FLAG_DOCUMENT_13",
        #                     "FLAG_DOCUMENT_14",
        #                     "FLAG_DOCUMENT_15",
        #                     "FLAG_DOCUMENT_16",
        #                     "FLAG_DOCUMENT_17",
        #                     "FLAG_DOCUMENT_18",
        #                     "FLAG_DOCUMENT_19",
        #                     "FLAG_DOCUMENT_2",
        #                     "FLAG_DOCUMENT_20",
        #                     "FLAG_DOCUMENT_21",
        #                     "FLAG_DOCUMENT_3",
        #                     "FLAG_DOCUMENT_4",
        #                     "FLAG_DOCUMENT_5",
        #                     "FLAG_DOCUMENT_6",
        #                     "FLAG_DOCUMENT_7",
        #                     "FLAG_DOCUMENT_8",
        #                     "FLAG_DOCUMENT_9",
        #                     "FLAG_EMAIL",
        #                     "FLAG_EMP_PHONE",
        #                     "FLAG_MOBIL",
        #                     "FLAG_OWN_CAR",
        #                     "FLAG_OWN_REALTY",
        #                     "FLAG_PHONE",
        #                     "FLAG_WORK_PHONE",
        #                     "FONDKAPREMONT_MODE",
        #                     "HOUR_APPR_PROCESS_START",
        #                     "HOUSETYPE_MODE",
        #                     "INCOME_CREDIT_PERC",
        #                     "INCOME_PER_PERSON",
        #                     "INSTAL_AMT_INSTALMENT_MAX",
        #                     "INSTAL_AMT_INSTALMENT_MEAN",
        #                     "INSTAL_AMT_INSTALMENT_SUM",
        #                     "INSTAL_AMT_PAYMENT_MAX",
        #                     "INSTAL_AMT_PAYMENT_MEAN",
        #                     "INSTAL_AMT_PAYMENT_MIN",
        #                     "INSTAL_AMT_PAYMENT_SUM",
        #                     "INSTAL_COUNT",
        #                     "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
        #                     "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
        #                     "INSTAL_DAYS_ENTRY_PAYMENT_SUM",
        #                     "INSTAL_DBD_MAX",
        #                     "INSTAL_DBD_MEAN",
        #                     "INSTAL_DBD_SUM",
        #                     "INSTAL_DPD_MAX",
        #                     "INSTAL_DPD_MEAN",
        #                     "INSTAL_DPD_SUM",
        #                     "INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE",
        #                     "INSTAL_PAYMENT_DIFF_MAX",
        #                     "INSTAL_PAYMENT_DIFF_MEAN",
        #                     "INSTAL_PAYMENT_DIFF_SUM",
        #                     "INSTAL_PAYMENT_DIFF_VAR",
        #                     "INSTAL_PAYMENT_PERC_MAX",
        #                     "INSTAL_PAYMENT_PERC_MEAN",
        #                     "INSTAL_PAYMENT_PERC_SUM",
        #                     "INSTAL_PAYMENT_PERC_VAR",
        #                     "LIVE_CITY_NOT_WORK_CITY",
        #                     "LIVE_REGION_NOT_WORK_REGION",
        #                     "NAME_CONTRACT_TYPE",
        #                     "NAME_EDUCATION_TYPE",
        #                     "NAME_FAMILY_STATUS",
        #                     "NAME_HOUSING_TYPE",
        #                     "NAME_INCOME_TYPE",
        #                     "NAME_TYPE_SUITE",
        #                     "OBS_30_CNT_SOCIAL_CIRCLE",
        #                     "OBS_60_CNT_SOCIAL_CIRCLE",
        #                     "OCCUPATION_TYPE",
        #                     "ORGANIZATION_TYPE",
        #                     "PAYMENT_RATE",
        #                     "POS_COUNT",
        #                     "POS_MONTHS_BALANCE_MAX",
        #                     "POS_MONTHS_BALANCE_MEAN",
        #                     "POS_MONTHS_BALANCE_SIZE",
        #                     "POS_SK_DPD_DEF_MAX",
        #                     "POS_SK_DPD_DEF_MEAN",
        #                     "POS_SK_DPD_MAX",
        #                     "POS_SK_DPD_MEAN",
        #                     "PREV_AMT_ANNUITY_MAX",
        #                     "PREV_AMT_ANNUITY_MEAN",
        #                     "PREV_AMT_ANNUITY_MIN",
        #                     "PREV_AMT_APPLICATION_MAX",
        #                     "PREV_AMT_APPLICATION_MEAN",
        #                     "PREV_AMT_APPLICATION_MIN",
        #                     "PREV_AMT_CREDIT_MAX",
        #                     "PREV_AMT_CREDIT_MEAN",
        #                     "PREV_AMT_CREDIT_MIN",
        #                     "PREV_AMT_DOWN_PAYMENT_MAX",
        #                     "PREV_AMT_DOWN_PAYMENT_MEAN",
        #                     "PREV_AMT_DOWN_PAYMENT_MIN",
        #                     "PREV_AMT_GOODS_PRICE_MAX",
        #                     "PREV_AMT_GOODS_PRICE_MEAN",
        #                     "PREV_AMT_GOODS_PRICE_MIN",
        #                     "PREV_APP_CREDIT_PERC_MAX",
        #                     "PREV_APP_CREDIT_PERC_MEAN",
        #                     "PREV_APP_CREDIT_PERC_MIN",
        #                     "PREV_CNT_PAYMENT_MEAN",
        #                     "PREV_CNT_PAYMENT_SUM",
        #                     "PREV_DAYS_DECISION_MAX",
        #                     "PREV_DAYS_DECISION_MEAN",
        #                     "PREV_DAYS_DECISION_MIN",
        #                     "PREV_HOUR_APPR_PROCESS_START_MAX",
        #                     "PREV_HOUR_APPR_PROCESS_START_MEAN",
        #                     "PREV_HOUR_APPR_PROCESS_START_MIN",
        #                     "PREV_RATE_DOWN_PAYMENT_MAX",
        #                     "PREV_RATE_DOWN_PAYMENT_MEAN",
        #                     "PREV_RATE_DOWN_PAYMENT_MIN",
        #                     "REGION_POPULATION_RELATIVE",
        #                     "REGION_RATING_CLIENT",
        #                     "REGION_RATING_CLIENT_W_CITY",
        #                     "REG_CITY_NOT_LIVE_CITY",
        #                     "REG_CITY_NOT_WORK_CITY",
        #                     "REG_REGION_NOT_LIVE_REGION",
        #                     "REG_REGION_NOT_WORK_REGION",
        #                     "SK_ID_CURR",
        #                     "WALLSMATERIAL_MODE",
        #                     "WEEKDAY_APPR_PROCESS_START",
        #                     "index"
        #                     ],
        #                     "index": [0],
        #                     "data": data_x
        #             }
        #           }

        # body_f = str.encode(json.dumps(data_f))
        # st.write(body_f)

        
        res = prediction(data_x)
        #st.write(res)

        score = 100 * round(float(res), 3)
        threshold = 100 * 0.769

        st.subheader(f"Score : {score}/100")
        st.subheader(f"Seuil minimum pour le score : {threshold}/100")

        if score >= threshold:
            st.subheader("Félicitations, prêt accordé!")
        else:
            st.subheader("Désolé, prêt refusé!")


if __name__ == "__main__":
    main()