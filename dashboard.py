import streamlit as st
import json
import requests
import pandas as pd

FastAPI_URL = 'http://127.0.0.1:8000/'#local URL

df = pd.read_csv('df_sample_for_dashboard.csv')
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)

def post_prediction(idClient: int):
    df_client = df.loc[df['index'] == idClient]
    dict_client = df_client.to_dict('records')[0]
    json_client = json.dumps(dict_client)
    #st.write(json_client)// pour debuggage
    #URL = FastAPI_URL + 'prediction/'// pour debuggage
    #st.write(URL) // pour debuggage
    response = requests.post(FastAPI_URL + 'prediction/', data = json_client)
    proba = eval(response.content)["probability"]
    return proba

def get_threshold():
    response = requests.get(FastAPI_URL + 'threshold/')
    return round(float(response.content), 3)

def main():

    st.set_page_config(page_title='Dashboard Application Crédit',
                    layout='centered',
                    initial_sidebar_state='expanded')

    st.title("Application Crédit")

    with st.sidebar:
        idClient = st.selectbox(label = 'Choisir un client', options = df['index'], key='idClient')

    st.write("L'ID du client sélectionné est: ", idClient)


    if st.button('Prédire'):
        res = post_prediction(idClient)
        score = 100 * round(float(res), 3)
        threshold = 100 * get_threshold()

        st.subheader(f"Score : {score}/100")
        st.subheader(f"Seuil minimum pour le score : {threshold}/100")

        if score >= threshold:
            st.subheader("Félicitations, prêt accordé!")
        else:
            st.subheader("Désolé, prêt refusé!")


if __name__ == "__main__":
    main()