import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numerize.numerize import numerize
import plotly.express as px
import plotly.graph_objs as go
import json
import requests
import time
from sklearn.preprocessing import StandardScaler

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    footer:after {
        content:'Made in KU with ❤️.'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Timeseries Anomaly Detection',
                           ['Fraud Detection', 'Assignment Overview', 'Team Members', 'References'],
                           icons=['search', 'folder', 'person', 'link'],
                           default_index=2)

# Fraud Detection Prediction Page
if selected == 'Fraud Detection':
    header_left, header_mid, header_right = st.columns([1, 12, 1])
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>CreditCard Fraud</h3>", unsafe_allow_html=True)

    file = st.file_uploader('Upload your creditcard.csv here', type='csv')
    data = pd.DataFrame(columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                                 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'])

    if file is not None:
        data = pd.read_csv(file)

    endpoint = 'https://ccfraudapi.onrender.com/ccmodel_prediction'

    # Initialize StandardScaler object
    scaler = StandardScaler()

    # Select columns to scale
    columns_to_scale = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    if data.empty or data.isnull().values.any():
        st.error("Error: Empty or null dataframe.")
    elif set(columns_to_scale).issubset(set(data.columns)):
        scaled_columns = scaler.fit_transform(data[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_columns, columns=[f'scaled_{col}' for col in columns_to_scale])
        df = pd.concat([data, scaled_df], axis=1).drop(columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class'])

        results = pd.DataFrame(columns=['rl_Time', 'rl_Amount', 'Class'])

        for i in range(len(df)):
            features = df.iloc[i].values.tolist()
            input_data_for_model = {f'scaled_{col}': features[j] for j, col in enumerate(columns_to_scale)}
            input_data_for_model_rl = {'rl_Time': data.iloc[i]['Time'], 'rl_Amount': data.iloc[i]['Amount']}

            input_json = json.dumps(input_data_for_model)
            response = requests.post(endpoint, data=input_json)
            predicted_class = response.json()

            results = results.append({**input_data_for_model_rl, "Class": predicted_class}, ignore_index=True)
            
            if i == 300:
                break

        non_fraud = results['Class'] == 0
        fraud = results['Class'] == 1
        Classes = results['Class'].value_counts()

        total_class = float(Classes.sum())
        total_non_fraud = float(non_fraud.sum())
        total_fraud = float(fraud.sum())
        total_Amount = float(results['rl_Amount'].sum())

        total1, total2, total3, total4 = st.columns(4)
        with total1:
            st.image('images/fraud-detection.png', width=None)
            st.metric(label='Total Class', value=numerize(total_class))
        with total2:
            st.image('images/non_fraud.png', width=160)
            st.metric(label='Total Non Fraud', value=numerize(total_non_fraud))
        with total3:
            st.image('images/fraud.png', width=158)
            st.metric(label='Total Fraud', value=numerize(total_fraud, 2))
        with total4:
            st.image('images/amount.png', width=200)
            st.metric(label='Total Amount', value=numerize(total_Amount))

        Q1, Q2 = st.columns(2)
        with Q1:
            fig1 = px.histogram(results, x='rl_Amount', color='Class', nbins=2, range_x=[0, 5000], title='Distribution of Amount for Class 0 and 1')
            fig1.update_layout(title={'x': 0.5}, plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        with Q2:
            fig2 = px.box(results, x='Class', y='rl_Time', points='all', title='Boxplot of Time for Class 0 and 1')
            fig2.update_layout(title={'x': 0.5}, plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

        Q3, Q4 = st.columns(2)
        with Q3:
            fig3 = px.scatter(results, x='rl_Time', y='rl_Amount', color='Class', title='Scatter Plot of Time vs Amount for Fraudulent and Non-Fraudulent Transactions')
            fig3.update_traces(marker=dict(size=10))
            fig3.update_layout(title={'x': 0.5}, plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
        with Q4:
            fraudulent_count = results[results['Class'] == 1].groupby(results['rl_Time'] // 3600)['Class'].count()
            non_fraudulent_count = results[results['Class'] == 0].groupby(results['rl_Time'] // 3600)['Class'].count()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=fraudulent_count.index, y=fraudulent_count.values, mode='lines', name='Fraudulent', line=dict(color='red')))
            fig4.add_trace(go.Scatter(x=non_fraudulent_count.index, y=non_fraudulent_count.values, mode='lines', name='Non-Fraudulent', marker=dict(color='blue', size=8)))
            fig4.update_layout(title='Time Series Plot of Fraudulent and Non Fraudulent Transactions', xaxis_title='Time (in hours)', yaxis_title='Count', plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

        st.table(results)

    with st.sidebar:
        Class_filter = st.multiselect(label='Select The Class', options=results['Class'].unique(), default=results['Class'].unique())
        Amount_filter = st.multiselect(label='Select Amount', options=results['rl_Amount'].unique(), default=results['rl_Amount'].unique())
        df1 = results.query('Class == @Class_filter & rl_Amount == @Amount_filter')

if selected == 'Assignment Overview':
    header_left, header_mid, header_right = st.columns([1, 12, 1])
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>Assignment Overview</h3>", unsafe_allow_html=True)
        
    st.header('Introduction')
    st.markdown("<p style='text-align: left; color: white;'>According to Nilson’s report in 2022, there are approximately 468 billion credit card transactions, including consumer, small business, and commercial credit cards, worth $3.789 trillion in transaction volume. Hence, it’s essential to understand how the fraud detection process works. Fraud detection is a process that uses machine learning algorithms to identify potentially fraudulent credit card transactions. The goal is to prevent unauthorized and illegal activities while minimizing the impact on legitimate transactions. Credit card fraud detection models analyze transaction data in real-time, looking for patterns and anomalies that may indicate fraudulent behavior. The primary objectives of these models are:</p>", unsafe_allow_html=True)
    st.markdown("<ul><li style='text-align: left; color: white;'>Accuracy</li><li style='text-align: left; color: white;'>Real-time Detection</li><li style='text-align: left; color: white;'>Scalability</li><li style='text-align: left; color: white;'>Adaptability</li></ul>", unsafe_allow_html=True)
    
    st.header('Step Involved in Detecting Anomalies')
    st.markdown("<ul><li style='text-align: left; color: white;'>Data Collection</li><li style='text-align: left; color: white;'>Data Preparation</li><li style='text-align: left; color: white;'>Model Training</li><li style='text-align: left; color: white;'>Feature Selection</li><li style='text-align: left; color: white;'>Model Selection</li><li style='text-align: left; color: white;'>Training and Validation</li><li style='text-align: left; color: white;'>Deployment</li><li style='text-align: left; color: white;'>Monitoring and Updates</li></ul>", unsafe_allow_html=True)

    st.header('Conclusion')
    st.markdown("<p style='text-align: left; color: white;'>Fraud detection systems play a crucial role in protecting consumers, financial institutions, and businesses from the detrimental effects of credit card fraud. These systems leverage advanced machine learning and data analysis techniques to identify suspicious transactions in real time, ensuring the integrity of financial transactions. By continuously adapting to evolving fraud patterns, fraud detection models contribute to a safer and more secure financial ecosystem for all stakeholders involved.</p>", unsafe_allow_html=True)

if selected == 'Team Members':
    header_left, header_mid, header_right = st.columns([1, 12, 1])
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>Team Members</h3>", unsafe_allow_html=True)
    
    st.header('Names')
    st.markdown("<ul><li style='text-align: left; color: white;'>Group Leader: ******</li><li style='text-align: left; color: white;'>Scrum Master: ******</li><li style='text-align: left; color: white;'>Team Member: ******</li></ul>", unsafe_allow_html=True)

if selected == 'References':
    header_left, header_mid, header_right = st.columns([1, 12, 1])
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>References</h3>", unsafe_allow_html=True)
    
    st.header('References')
    st.markdown("<ul><li style='text-align: left; color: white;'><a href='https://realpython.com' style='color: red;'>Real Python</a></li><li style='text-align: left; color: white;'><a href='https://w3schools.com' style='color: red;'>W3Schools</a></li><li style='text-align: left; color: white;'><a href='https://towardsdatascience.com' style='color: red;'>Towards Data Science</a></li></ul>", unsafe_allow_html=True)
