import ast
from PIL import Image
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import seaborn as sns
import shap
from shap.plots import waterfall
import streamlit as st
import urllib.request
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='Dashboard Home Loan App',
                layout='centered',
                initial_sidebar_state='expanded')

logo = Image.open('logo_pret_a_depenser.png')

def feature_distribution_bar_chart(dataframe, feature, row_index):
    # Tracé de l'histogramme
    fig, ax = plt.subplots(figsize = (5,3))
    data = dataframe[feature]

    # Extract the row of the dataframe
    row = dataframe.loc[dataframe['index'] == row_index]
    
    # Extract the row of the dataframe
    row_value = float(row[feature])

    # Annotations (on tracer l'histogramme mais c'est juste pour récupérer la valeur ymax)
    y, x, _ = plt.hist(data)
    ymax = y.max()
    ax.text(row_value, ymax/2, " ← Customer " + str(row_index), size = 10, alpha = 1, color = 'blue')

    # Tracé des pourcentiles en rouge
    ax.axvline(row_value, color='blue', linestyle = "--")

    # Tracé de l'histogramme (pour écraser le 1er tracé de l'histogramme plus haut)
    plt.hist(data, color = "skyblue", ec="white") # Crée l'histogramme
    plt.title(feature)
    fig = plt.show() # Affiche l'histogramme
    return fig

def feature_distribution_boxplot(dataframe, feature, row_index):
    # Extract the row of the dataframe
    row = dataframe.loc[dataframe['index'] == row_index]
    
    # Extract the row of the dataframe
    row_value = float(row[feature])

    fig, ax = plt.subplots(figsize = (5,3))

    # Tracé du boxplot
    plt.xticks(rotation=90)
    data = dataframe[feature]
    min_raw = round(dataframe[feature].min())
    min_r = round(min_raw, abs(1 - (len(str(min_raw)))))
    max_raw = round(dataframe[feature].max())
    max_r = round(max_raw, abs(1 - (len(str(max_raw)))))
    step = (max_r - min_r) / 20
    if (step != 0):
        plt.xticks(np.arange(min_r, max_r, step))
        red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='black')
        mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='black')
        
        ax = sns.boxplot(x = data, orient="h", color='skyblue', flierprops=red_circle, showmeans=True, meanprops=mean_shape)
        ax.set_title(feature)
        if feature == 'PAYMENT RATE':
            ax.text(row_value, 1.04, "Customer " + str(row_index) + "\n↓", size=10, ha="center", color='blue')
        else:
            ax.text(row_value, -0.02, "Customer " + str(row_index) + "\n↓", size=10, ha="center", color='blue')
    else:
        red_circle = dict(markerfacecolor='red', marker='o')
        mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='black')
        
        plt.boxplot(x=dataframe[feature], vert=False, flierprops=red_circle, 
             showmeans=True, meanprops=mean_shape)
        plt.title(feature)
        if feature == 'PAYMENT RATE':
            plt.text(row_value, 1.04, "Customer " + str(row_index) + "\n↓", size=10, ha="center", color='blue')
        else:
            plt.text(row_value, -0.02, "Customer " + str(row_index) + "\n↓", size=10, ha="center", color='blue')
    fig = plt.show() # Affiche le boxplot
    return fig

# make any grid with a function
def make_grid(cols, row): #cols and row variable names are mixed up
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(row)
    return grid

# Load the serialized explanation object from the saved file
# with open('lgbm_opti_class_weight_explainer_sample.pkl', 'rb') as f:
#     explainer = pickle.load(f)
with open('shap_values.pickle', 'rb') as f:
    shap_values = pickle.load(f)

df = pd.read_csv('df_valid_tt_sample.csv', index_col = 0)
df.drop('TARGET', axis=1, inplace=True)
index_column = df['index']

df_info_raw = df[['index', 'CODE_GENDER', 'DAYS_BIRTH',  'AMT_INCOME_TOTAL', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'PAYMENT_RATE']].copy()

df_info_polished = df_info_raw.copy()
df_info_polished = df_info_polished.rename(columns={
    'CODE_GENDER': 'GENDER',
    'DAYS_BIRTH': 'AGE',
    'AMT_INCOME_TOTAL': 'INCOME',
    'EXT_SOURCE_2': 'SCORE 2',
    'EXT_SOURCE_3': 'SCORE 3',
    'AMT_CREDIT': 'CREDIT AMOUNT',
    'PAYMENT_RATE': 'PAYMENT RATE'
})
df_info_polished['GENDER'] = df_info_polished['GENDER'].replace({1: 'Male', 0: 'Female'})
df_info_polished.insert(loc=df_info_polished.columns.get_loc('GENDER') + 1, column='AGE_YEARS', value=round(df_info_polished['AGE'] / -365.25))
df_info_polished.drop('AGE', axis=1, inplace=True)
df_info_polished['AGE_YEARS'] = df_info_polished['AGE_YEARS'].round(1)

df_shap_values = pd.DataFrame()
df_shap_values['index_number'] = df_info_polished['index']
df_shap_values['shap_value_index'] = 0
df_shap_values.loc[:,'shap_value_index'] = np.arange(1, 101)

def get_shape_values_index(client_index_value):
    shape_values_index = df_shap_values[df_shap_values['index_number'] == client_index_value]['shap_value_index'].values[0]
    return shape_values_index


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

    #url = 'http://127.0.0.1:5000'# API Endpoint - local machine
    url = 'https://ocr-p7-api-mlflow-proba-qljnp.francecentral.inference.ml.azure.com/score' #API Endpoint in the cloud (Azure) opti proba weight class

    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = 'GBzKoAD0Bpc8rYUHS4iDwbHJrdwYCl4P' #Key for Predict_proba Model opti weight class
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'lgbm-opti-class-weight-proba-2' }

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

st.sidebar.image(logo, width=240, use_column_width='always')


def main():

    st.title("🏠 Application for Home Loan")

    with st.sidebar:
        idClient = st.selectbox(label = '👇 Select a customer ID', options = index_column, key='idClient')
    
    st.markdown(f"➡️ The ID of the selected customer is : <span style='color: dodgerblue'>{idClient}</span>", unsafe_allow_html=True)
    data_client = df.loc[df['index'] == idClient]
    display_client = df_info_polished.loc[df_info_polished['index'] == idClient].round({'AGE_YEARS': 0})
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(display_client.style.format({"AGE_YEARS": "{:.0f}", "INCOME": "{:,.0f}",
         "SCORE 2": "{:.5f}",  "SCORE 3": "{:.5f}", "CREDIT AMOUNT": "{:,.0f}", "PAYMENT RATE": "{:.5f}"}))

    if st.sidebar.checkbox("🔮 Predict", key=38):
        data_x = np.asarray(data_client).tolist()
        
        res = prediction(data_x)

        score = 100 * round(res, 3)
        threshold = 100 * 0.159

        if score <= threshold:
            st.markdown(f'<p style="color: green;">🟢 Congratulations, Loan is granted!<br> Your score : {round(score, 1)} is below the threshold : {round(threshold, 1)}.</p>',
            unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: red;">⛔️ Sorry, loan is not granted.<br> Your score : {round(score, 1)} is above the threshold : {round(threshold, 1)}.</p>',
            unsafe_allow_html=True)

    if st.sidebar.checkbox("ℹ️ Explain prediction", key=25):
        #Display the SHAP values for the data point in a Streamlit app
        st.write("⬇️ Below are the parameters who have the most impact on the decision:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # fig = shap.plots.waterfall(explainer(df.loc[df['index'] == idClient])[0])
        # st.pyplot(fig)

        shape_value_index = get_shape_values_index(idClient)
        fig = shap.plots.waterfall(shap_values[shape_value_index - 1])
        st.pyplot(fig)

    if st.sidebar.checkbox("🌐 Compare with other customers", key=42):
        st.write("⬇️ Below are the comparison charts with other customers:")

        grid = make_grid(7, 2)

        grid[0][0].write('<span style="font-weight:bold; font-size:18px; color:blue;">Bar charts</span>', unsafe_allow_html=True)
        grid[0][1].write('<span style="font-weight:bold; font-size:18px; color:blue;">Boxplots</span>', unsafe_allow_html=True)

        fig1 = feature_distribution_bar_chart(df_info_polished, 'AGE_YEARS', idClient)
        grid[1][0].pyplot(fig1)
        fig7 = feature_distribution_boxplot(df_info_polished, 'AGE_YEARS', idClient)
        grid[1][1].pyplot(fig7)

        fig2 = feature_distribution_bar_chart(df_info_polished, 'INCOME', idClient)
        grid[2][0].pyplot(fig2)
        fig8 = feature_distribution_boxplot(df_info_polished, 'INCOME', idClient)
        grid[2][1].pyplot(fig8)

        fig3 = feature_distribution_bar_chart(df_info_polished, 'SCORE 2', idClient)
        grid[3][0].pyplot(fig3)
        fig9 = feature_distribution_boxplot(df_info_polished, 'SCORE 2', idClient)
        grid[3][1].pyplot(fig9)

        fig4 = feature_distribution_bar_chart(df_info_polished, 'SCORE 3', idClient)
        grid[4][0].pyplot(fig4)
        fig10 = feature_distribution_boxplot(df_info_polished, 'SCORE 3', idClient)
        grid[4][1].pyplot(fig10)

        fig5 = feature_distribution_bar_chart(df_info_polished, 'CREDIT AMOUNT', idClient)
        grid[5][0].pyplot(fig5)
        fig11 = feature_distribution_boxplot(df_info_polished, 'CREDIT AMOUNT', idClient)
        grid[5][1].pyplot(fig11)

        fig6 = feature_distribution_bar_chart(df_info_polished, 'PAYMENT RATE', idClient)
        grid[6][0].pyplot(fig6)
        fig12 = feature_distribution_boxplot(df_info_polished, 'PAYMENT RATE', idClient)
        grid[6][1].pyplot(fig12)
    

if __name__ == "__main__":
    main()