
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numerize.numerize import numerize
import plotly.express as px
import plotly.graph_objs as go
import json
import requests
import pandas as pd
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
	        #background-color: red;
	        padding: 5px;
	        top: 2px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Timeseries Anomaly Detection',
                          
                          ['Fraud Detection',
                           'Assignment Overview',
                           'Team Members',
                           'References'],
                          icons=['search','folder','person','link'],
                          default_index=2)
    
    
# Fraud Detection Prediction Page
if (selected == 'Fraud Detection'):
    
    header_left,header_mid,header_right = st.columns([1,12,1])
    # page title
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>CreditCard Fraud</h3>", unsafe_allow_html=True)

    
    # getting the input data from the user
    
    file = st.file_uploader('Upload your creditcard.csv here', type='csv')
    
    data=pd.DataFrame(columns=['Time (second)','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
                                  'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class'])
   
    if file is not None:
        # read data from file
        data = pd.read_csv(file)
    
    endpoint = 'https://ccfraudapi.onrender.com/ccmodel_prediction'

    
    
    data.columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
        'Class']
    
    # Initialize StandardScaler object
    scaler = StandardScaler()
    scaled_columns =[]

    # Select columns to scale
    columns_to_scale = ['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    # Check if the dataframe is empty or null
    if data.empty or data.isnull().values.any():
        print("Error: Empty or null dataframe.")
    else:
        # Check if the columns exist in the dataframe
        if set(columns_to_scale).issubset(set(data.columns)):
            # Scale the selected columns
            scaled_columns = scaler.fit_transform(data[columns_to_scale])
            
        else:
            print("Error: Selected columns do not exist in the dataframe.")
            
    
    # Scale the selected columns
    #scaled_columns = scaler.fit_transform(data[columns_to_scale])

    # Create new DataFrame with the scaled columns
    scaled_df = pd.DataFrame(scaled_columns, columns=[f'scaled_{col}' for col in columns_to_scale])

    # Append the new DataFrame to the original DataFrame
    df = pd.concat([data, scaled_df], axis=1)

    # create a copy of the data and scale the 30 main columns
    df = df.drop(columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Class'])
    print(df.head(3))

    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=['rl_Time', 'rl_Amount',
        'Class'])


    # Loop over each row in the dataset
    for i in range(len(df)):
        
        # Get the features for the current row
        features = df.iloc[i].values.tolist()
        print(features)
        input_data_for_model = {
        'Time' : features[2],
        'V1' : features[3],
        'V2' : features[4],
        'V3' : features[5],
        'V4' : features[6],
        'V5' : features[7],
        'V6' : features[8],
        'V7' : features[9],
        'V8' : features[10],
        'V9' : features[11],
        'V10' : features[12],
        'V11' : features[13],
        'V12' : features[14],
        'V13' : features[15],
        'V14' : features[16],
        'V15' : features[17],
        'V16' : features[18],
        'V17' : features[19],
        'V18' : features[20],
        'V19' : features[21],
        'V20' : features[22],
        'V21' : features[23],
        'V22' : features[24],
        'V23' : features[25],
        'V24' : features[26],
        'V25' : features[27],
        'V26' : features[28],
        'V27' : features[29],
        'V28' : features[30],
        'Amount' : features[31]
        }
        
        input_data_for_model_rl= {
        'rl_Time' : features[0],
        'rl_Amount' : features[1]
        }
        print(input_data_for_model)
        input_json = json.dumps(input_data_for_model)

        response = requests.post(endpoint, data=input_json)
        # Get the predicted class from the API response
        predicted_class = response.json()
        #pd.Series(input_data_for_model)
        print(predicted_class)
        #results = pd.concat([data, pd.Series(predicted_class, name="Class")], axis=1)    
        results = results.append({**input_data_for_model_rl, "Class": predicted_class}, ignore_index=True)
        print(results.head(i))
        #results = pd.DataFrame(columns=['Time', 'Amount', 'Class']) 
        non_fraud= results['Class'] ==0
        fraud= results['Class'] ==1 
        Classes= results['Class'].value_counts()
        placeholder = st.empty()
        placeholder.empty()
        
        im=st.empty()
        a=st.empty()
        ims=st.empty()
        b=st.empty()
        imr=st.empty()
        c=st.empty()
        imm=st.empty()
        d=st.empty()
        qq1=st.empty()
        qq2=st.empty()
        qq3=st.empty()
        qq4=st.empty()
        im.empty()
        a.empty()
        ims.empty()
        b.empty()
        imr.empty()
        c.empty()
        imm.empty()
        d.empty()
        qq1.empty()
        qq2.empty()
        qq3.empty()
        qq4.empty()
        #placeholder.table(results)
        non_fraud= results['Class'] ==0
        fraud= results['Class'] ==1 
        Classes= results['Class'].value_counts()
      
        
        non_fraud= results['Class'] ==0
        fraud= results['Class'] ==1 
        Classes= results['Class'].value_counts()
      
        
    


        

    
        total_class = float(Classes.sum())
        
        total_non_fraud = float(non_fraud.sum())
        total_fraud = float(fraud.sum())
        total_Amount= float(results['rl_Amount'].sum()) 
        
        ck=st.empty()
        total1,total2,total3,total4 = ck.columns(4)

        with total1:
            im=st.empty()
            a=st.empty()
            im.image('images/fraud-detection.png',width=None)
            a.metric(label = 'Total Class', value= numerize(total_class))
           
            
        with total2:
            ims=st.empty()
            b=st.empty()
            ims.image('images/non_fraud.png',width=160)
            b.metric(label='Total Non Fraud', value=numerize(total_non_fraud))
           

        with total3:
            imr=st.empty()
            c=st.empty()
            imr.image('images/fraud.png',width=158)
            c.metric(label= 'Total Fraud',value=numerize(total_fraud,2))
            
            

        with total4:
            imm=st.empty()
            d=st.empty()
            imm.image('images/amount.png',width=200)
            d.metric(label='Total Amount',value=numerize(total_Amount))
            

        
        qqq=st.empty()
        Q1,Q2 = qqq.columns(2)

        with Q1:
            fig1 = px.histogram(results, x='rl_Amount', color='Class', nbins=2, range_x=[0, 5000], title='Distribution of Amount for Class 0 and 1')
            fig1.update_layout(title = {'x' : 0.5},
                                            plot_bgcolor = "rgba(0,0,0,0)",
                                            xaxis =(dict(showgrid = False)),
                                            yaxis =(dict(showgrid = False)))
            qq1=st.empty()
            qq1.plotly_chart(fig1, theme="streamlit", use_container_width=True)
            
            

        with Q2:
            
            #Plot 2: Boxplot of Time for Class 0 and 1
            fig2 = px.box(results, x='Class', y='rl_Time', points='all', title='Boxplot of Time for Class 0 and 1')
            fig2.update_layout(title = {'x' : 0.5},
                                            plot_bgcolor = "rgba(0,0,0,0)",
                                            xaxis =(dict(showgrid = False)),
                                            yaxis =(dict(showgrid = False)))
            qq2=st.empty()
            qq2.plotly_chart(fig2, theme="streamlit", use_container_width=True)
            
            
        
        qqqq=st.empty()
        Q3,Q4 = qqqq.columns(2)
        
        with Q3:
            
        
            # Create scatter plot
            fig3 = px.scatter(results, x='rl_Time', y='rl_Amount', color='Class',
                            title='Scatter Plot of Time vs Amount for Fraudulent and Non-Fraudulent Transactions')
            fig3.update_traces(marker=dict(size=10))
            fig3.update_layout(title = {'x' : 0.5},
                                            plot_bgcolor = "rgba(0,0,0,0)",
                                            xaxis =(dict(showgrid = False)),
                                            yaxis =(dict(showgrid = False)))
            qq3=st.empty()
            qq3.plotly_chart(fig3, theme="streamlit", use_container_width=True)

            
            
            
        with Q4:
            
            # Select data for fraudulent and non-fraudulent transactions
            fraudulent = results[results['Class'] == 1]
            non_fraudulent = results[results['Class'] == 0]
            # Group data by hour and count the number of fraudulent transactions
            fraudulent_count = fraudulent.groupby(fraudulent['rl_Time'].apply(lambda x: int(x/3600)))['Class'].count()
            non_fraudulent_count = non_fraudulent.groupby(non_fraudulent['rl_Time'].apply(lambda x: int(x/3600)))['Class'].count()

            # Create time series plot
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=fraudulent_count.index, y=fraudulent_count.values,
                                    mode='lines', name='Fraudulent', line=dict(color='red')))
            fig4.add_trace(go.Scatter(x=non_fraudulent_count.index, y=non_fraudulent_count.values,
                                    mode='lines', name='Non-Fraudulent', marker=dict(color='blue', size=8)))
            fig4.update_layout(title='Time Series Plot of Fraudulent and Non Fraudulent Transactions',
                            xaxis_title='Time (in hours)', yaxis_title='Count', plot_bgcolor = "rgba(0,0,0,0)",
                                            xaxis =(dict(showgrid = False)),
                                            yaxis =(dict(showgrid = False)))
            qq4=st.empty()
            qq4.plotly_chart(fig4, theme="streamlit", use_container_width=True)
            
        
        mk=st.empty()
        
        cm=mk.container()
        cm.table(results) 
         
        
        
        
            
        if i == 300:
            
            break
        time.sleep(2)
        placeholder.empty()
        im.empty()
        a.empty()
        ims.empty()
        b.empty()
        imr.empty()
        c.empty()
        imm.empty()
        d.empty()
        qq1.empty()
        qq2.empty()
        qq3.empty()
        qq4.empty()
        ck.empty()
        qqq.empty()
        qqqq.empty()
         
        mk.empty()
        
        
            
            # Add the features and predicted class to the results DataFrame
            #results = results.concat({**input_data_for_model, "Class": predicted_class}, ignore_index=True)
            
            
            
            
        # Print the results for this iteration
    
        
       
    
        
    # non_fraud= results['Class'] ==0
    # fraud= results['Class'] ==1 
    # Classes= results['Class'].value_counts()
      
    
    # st.table(results.head())
    


    with st.sidebar:
        
        
        Class_filter = st.multiselect(label= 'Select The Class',options=results['Class'].unique(),default=results['Class'].unique())

        Amount_filter = st.multiselect(label='Select Amount',options=results['rl_Amount'].unique(),default=results['rl_Amount'].unique())
        
    
    # # merge the 1D dataframe to the multi-dimensional dataframe using concat
    

    # # merge the 1D dataframe to the multi-dimensional dataframe using join
    # #merged_df = df.join(series)

       

    df1 = results.query('Class == @Class_filter & rl_Amount == @Amount_filter')
    
    # total_class = float(Classes.sum())
    
    # total_non_fraud = float(non_fraud.sum())
    # total_fraud = float(fraud.sum())
    # total_Amount= float(df1['rl_Amount'].sum()) 
    
    # total1,total2,total3,total4 = st.columns(4)

    # with total1:
    #     st.image('images/fraud-detection.png',width=None)
    #     st.metric(label = 'Total Class', value= numerize(total_class))
        
    # with total2:
    #     st.image('images/non_fraud.png',width=160)
    #     st.metric(label='Total Non Fraud', value=numerize(total_non_fraud))

    # with total3:
    #     st.image('images/fraud.png',width=158)
    #     st.metric(label= 'Total Fraud',value=numerize(total_fraud,2))

    # with total4:
    #     st.image('images/amount.png',width=200)
    #     st.metric(label='Total Amount',value=numerize(total_Amount))

    
    
    # Q1,Q2 = st.columns(2)

    # with Q1:
    #     fig1 = px.histogram(results, x='rl_Amount', color='Class', nbins=50, range_x=[0, 5000], title='Distribution of Amount for Class 0 and 1')
    #     fig1.update_layout(title = {'x' : 0.5},
    #                                     plot_bgcolor = "rgba(0,0,0,0)",
    #                                     xaxis =(dict(showgrid = False)),
    #                                     yaxis =(dict(showgrid = False)))
    #     st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        
        

    # with Q2:
        
    #     #Plot 2: Boxplot of Time for Class 0 and 1
    #     fig2 = px.box(results, x='Class', y='rl_Time', points='all', title='Boxplot of Time for Class 0 and 1')
    #     fig2.update_layout(title = {'x' : 0.5},
    #                                     plot_bgcolor = "rgba(0,0,0,0)",
    #                                     xaxis =(dict(showgrid = False)),
    #                                     yaxis =(dict(showgrid = False)))
    #     st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
        
        
       
       
    # Q3,Q4 = st.columns(2)
    
    # with Q3:
           
       
    #     # Create scatter plot
    #     fig3 = px.scatter(results, x='rl_Time', y='rl_Amount', color='Class',
    #                     title='Scatter Plot of Time vs Amount for Fraudulent and Non-Fraudulent Transactions')
    #     fig3.update_traces(marker=dict(size=3))
    #     fig3.update_layout(title = {'x' : 0.5},
    #                                     plot_bgcolor = "rgba(0,0,0,0)",
    #                                     xaxis =(dict(showgrid = False)),
    #                                     yaxis =(dict(showgrid = False)))
    #     st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

        
        
        
    # with Q4:
        
    #     # Select data for fraudulent and non-fraudulent transactions
    #     fraudulent = results[results['Class'] == 1]
    #     non_fraudulent = results[results['Class'] == 0]
    #     # Group data by hour and count the number of fraudulent transactions
    #     fraudulent_count = fraudulent.groupby(fraudulent['rl_Time'].apply(lambda x: int(x/3600)))['Class'].count()

    #     # Create time series plot
    #     fig4 = go.Figure()
    #     fig4.add_trace(go.Scatter(x=fraudulent_count.index, y=fraudulent_count.values,
    #                              mode='lines+markers', name='Fraudulent'))
    #     fig4.update_layout(title='Time Series Plot of Fraudulent Transactions',
    #                       xaxis_title='Time (in hours)', yaxis_title='Count', plot_bgcolor = "rgba(0,0,0,0)",
    #                                     xaxis =(dict(showgrid = False)),
    #                                     yaxis =(dict(showgrid = False)))
    #     st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

    
    
if (selected == 'Assignment Overview'):
    
    header_left,header_mid,header_right = st.columns([1,12,1])
    # page title
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>Assignment Overview</h3>", unsafe_allow_html=True)
        
    st.header('Introduction')
    st.markdown("<p style='text-align: left; color: white;'>According to Nilson’s report in 2022, there are approximately 468 billion credit card transactions, including consumer, small business, and commercial credit, debit, and prepaid products. These transactions will increase monthly as the permeation of internet technologies to semi-urban and rural areas is becoming rife, with a higher number of small businesses transitioning to digital payments due to the negative impact of the pandemic. As digital transactions are increasing, there is a potential for hackers with bad intent to carry out unauthorized transactions on the credit cards of consumers, individuals, and small businesses relying on the use of credit cards to meet their business needs. Financial service providers facilitating the execution of these transactions need to integrate with their system a solution that will automatically detect and classify transactions that do not reflect a known pattern based on the history of such transactions. </p>", unsafe_allow_html=True)
    st.subheader('Aims and Objectives of Time series Anomaly Detection ')
    st.markdown("<p style='text-align: left; color: white; '>This project aims to develop and implement a deep-learning solution with the ability to classify and detect credit card transactions in real time as either fraudulent or non-fraudulent. The objectives of this application are: </p>", unsafe_allow_html=True)
    st.markdown('<ol><li>Develop a deep neural network trained with each credit card transaction within the dataset as input.  </li><li>Develop a neural network that will be trained with the credit card transaction dataset as a time series input.</li><li>Develop a deep neural network trained with the credit card transaction data set as input and classify each transaction as either fraudulent or non-fraudulent based on the target in the dataset.</li><li>Develop a deep neural network trained to detect anomalous transactions and evaluate the performance of the deep neural networks on the validation dataset using metrics such as F1, precision, recall, accuracy, and confusion matrix. </li><li>Develop a rest API-based web application on which the best-performing deep neural network will be deployed for consumption to classify and detect anomalous credit card transactions.</li></ol>', unsafe_allow_html=True)
    

if (selected == 'Team Members'):
    
    header_left,header_mid,header_right = st.columns([1,12,1])
    # page title
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>Team Members</h3>", unsafe_allow_html=True)
    #st.markdown('<ol><li>Choudhary, Muskan  k2268186 </li><li>Acquaisie, Mark A  k2218627 </li><li>Arumalla, Srija  k2148736 </li><li>Olawale, Arowolo  K2265988</li></ol>', unsafe_allow_html=True)
    data_t = {
        'Name': ['Choudhary, Muskan', 'Acquaisie, Mark A', 'Arumalla, Srija', 'Olawale, Arowolo'],
        'K-Number': ['k2268186', 'k2218627', 'k2148736', 'K2265988']}

    team = pd.DataFrame(data_t, columns=['Name', 'K-Number'])
    st.table(team)
        
    




if (selected == 'References'):
    
    header_left,header_mid,header_right = st.columns([1,12,1])
    # page title
    with header_mid:
        st.title('Timeseries Anomaly Detection')
        st.markdown("<h3 style='text-align: center; color: red;'>References & Evaluation Metrics</h3>", unsafe_allow_html=True)
        st.subheader('References')
        st.markdown("<p style='text-align: left; color: white; '>Alarfaj, F.K. et al. (2022) “Credit card fraud detection using state-of-the-art machine learning and Deep Learning Algorithms,” IEEE Access, 10, pp. 39700–39715. Available at: https://doi.org/10.1109/access.2022.3166891.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; color: white; '>Fiore, U. et al. (2019) “Using generative adversarial networks for improving classification effectiveness in credit card fraud detection,” Information Sciences, 479, pp. 448–455. Available at: https://doi.org/10.1016/j.ins.2017.12.030.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; color: white; '>Forough, J. and Momtazi, S. (2021) “Sequential Credit Card Fraud Detection: A joint deep neural network and probabilistic graphical model approach,” Expert Systems, 39(1). Available at: https://doi.org/10.1111/exsy.12795. </p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; color: white; '>Global Network Cards - purchase transactions (2022) Nilson Report | Research | The World's Top Card Issuers and Merchant Acquirers. Available at: https://nilsonreport.com/research_featured_chart.php (Accessed: March 13, 2023).</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; color: white; '>Pumsirirat, A. and Yan, L. (2018) “Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine,” International Journal of Advanced Computer Science and Applications, 9(1). Available at: https://doi.org/10.14569/ijacsa.2018.090103.</p>", unsafe_allow_html=True)
        st.subheader('')
        st.subheader('Evaluation Metrics')
        st.markdown("<p style='text-align: left; color: white; '> Evaluated performance of the models on the test dataset with tabulated results below in a comparative study (example metrics: F1, precision, recall, accuracy,confusion matrix, etc.)</p>", unsafe_allow_html=True)
        data_t = {'Evaluation Metrics' : ['F1 Score', 'Precision', 'Recall', 'Accuracy'],
        'Model 1': ['0.9983', '0.9966', '1.0', '0.9982'],
        'Model 2': ['0.9988', '0.9976', '1.0', '0.9988'],
        'Model 3': ['0.9995', '0.9991', '1.0', '0.9995'],
        'Model 4': ['0.1792', '0.9757', '0.0986', '0.5470']}
        st.markdown("<p style='text-align: left; color: white; '> Please refer to our report for detailed insights and Confusion Matrix table</p>", unsafe_allow_html=True)
        
         
        Evaluation = pd.DataFrame(data_t, columns=['Evaluation Metrics', 'Model 1', 'Model 2', 'Model 3', 'Model 4'])
    st.table(Evaluation)
        

