#importing necessary packages
import numpy as np #for array operations
import pandas as pd #for creating and handling arrays
import matplotlib.pyplot #for visualizations
import seaborn as sns #for adv. visualizations
import streamlit as st
from streamlit_option_menu import option_menu #for option box


#1. DATA COLLECTION
data = pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (4)\Train.csv")

#2. DATA PRE-PROCESSING
State_Name_unique = data['Warehouse_block'].unique().tolist() #getting a list of all unique values in State_Name column
State_Name_index = np.arange(1,len(State_Name_unique)+1,1) #creating an array fo numbers as indexing with length of above list
State_Name_Next_to_num = {i:j for i,j in zip(State_Name_unique,State_Name_index)} #creating a dictionary for mapping text to num

#mapping text labels in State_Name columns to number
data['Warehouse_block'] =data['Warehouse_block'].map(State_Name_Next_to_num)

#displayig new updated dataframe
data.head(3)

State_Name_unique = data['Mode_of_Shipment'].unique().tolist() #getting a list of all unique values in State_Name column
State_Name_index = np.arange(1,len(State_Name_unique)+1,1) #creating an array fo numbers as indexing with length of above list
State_Name_Next_to_num = {i:j for i,j in zip(State_Name_unique,State_Name_index)} #creating a dictionary for mapping text to num

#mapping text labels in State_Name columns to number
data['Mode_of_Shipment'] =data['Mode_of_Shipment'].map(State_Name_Next_to_num)

#displayig new updated dataframe
data.head(3)

State_Name_unique = data['Product_importance'].unique().tolist() #getting a list of all unique values in State_Name column
State_Name_index = np.arange(1,len(State_Name_unique)+1,1) #creating an array fo numbers as indexing with length of above list
State_Name_Next_to_num = {i:j for i,j in zip(State_Name_unique,State_Name_index)} #creating a dictionary for mapping text to num

#mapping text labels in State_Name columns to number
data['Product_importance'] =data['Product_importance'].map(State_Name_Next_to_num)

#displayig new updated dataframe
data.head(3)

State_Name_unique = data['Gender'].unique().tolist() #getting a list of all unique values in State_Name column
State_Name_index = np.arange(1,len(State_Name_unique)+1,1) #creating an array fo numbers as indexing with length of above list
State_Name_Next_to_num = {i:j for i,j in zip(State_Name_unique,State_Name_index)} #creating a dictionary for mapping text to num

#mapping text labels in State_Name columns to number
data['Gender'] =data['Gender'].map(State_Name_Next_to_num)

#displayig new updated dataframe
data.head(3)

necessary_columns = ['Warehouse_block','Mode_of_Shipment','Customer_care_calls','Customer_rating','Cost_of_the_Product','Prior_purchases','Product_importance','Gender','Discount_offered','Weight_in_gms','Reached.on.Time_Y.N'] #creating a list of wanted columns
data_pda = data[necessary_columns] #updating the dataframe to contain only necessary columns
data_pda.head() #displaying new dataframe


with st.sidebar: #adding option menu to sidebar
    model_selection = option_menu('SELECT A MODEL',options=['DECISION TREE','RANDOM FOREST','XGBOOST','MODEL PARAMETERS'])

if model_selection == 'DECISION TREE':
    st.title("INTEGRATING GUI with ML MODELS")  # setting title of webpage

    # taking new user inputs
    warehouse = st.number_input('ENTER WAREHOUSE BLOCK', step=1, min_value=1, max_value=6)
    customer_care_calls = st.number_input('ENTER CUSTOMER CARE CALLS', step=1)
    customer_rating = st.number_input('ENTER CUSTOMER RATING', step=1, min_value=1, max_value=5)
    cost_of_the_product = st.number_input('ENTER COST OF THE PRODUCT', step=1)
    prior_purchases = st.number_input('ENTER PRIOR PURCHASES', step=1)
    product_importance = st.number_input('ENTER PRODUCT IMPORTANCE', step=1, min_value=1, max_value=3)
    gender = st.number_input('ENTER GENDER', step=1, min_value=1, max_value=2)
    discount_offered = st.number_input('ENTER DISCOUNT OFFERED', step=1)
    weight_in_gms = st.number_input('ENTER WEIGHT IN GRAMS', step=1)
    reached_on_Time_YN = st.number_input('ENTER REACHED ON TIME', step=1, min_value=0, max_value=1)

    # storing new user input into a 2d array
    new_user_input = [
        [warehouse, customer_care_calls, customer_rating, cost_of_the_product, prior_purchases, product_importance,
         gender, discount_offered, weight_in_gms, reached_on_Time_YN]]

    from sklearn.tree import DecisionTreeClassifier  # importing the algo

    # segregating data into features and target
    x_columns = ['Warehouse_block', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
                 'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms',
                 'Reached.on.Time_Y.N']  # definig input columns
    x = data_pda[x_columns].values  # choosing input columns

    y = data_pda['Mode_of_Shipment'].values  # choosing output column

    # splitting data into training and testing partitions
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2800, test_size=0.1)

    # 3. MODEL TRAINING
    from sklearn.tree import DecisionTreeClassifier  # importing the algo

    dt_model = DecisionTreeClassifier()  # initializing the model
    dt_model.fit(x_train, y_train)

    submit_button = st.button("SUBMIT")
    if submit_button:
        # making the model predict answeers for x_test
        y_pred = dt_model.predict(x_test)

        # defining actual answers
        y_actual = y_test
        # making
        st.subheader('MODEL DIAGNOSIS')
        dt_model_output = dt_model.predict(new_user_input)
        if dt_model_output[0] == 1:
            st.error('THE SHIPMENT MODE IS FLIGHT')
        if dt_model_output[0] == 2:
            st.success('THE SHIPMENT MODE IS SHIP')
        if dt_model_output[0] == 3:
            st.success('THE SHIPMENT MODE IS ROAD')
        st.subheader('MODEL PARAMETER')

        # comparing actual answers with predictions from the model
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix,classification_report
        accuracy = accuracy_score(y_pred, y_actual)
        st.success(f'accuracy : {accuracy}')


        from sklearn.metrics import classification_report
        st.subheader('CLASSIFICATION REPORT')
        classif = classification_report(y_actual, y_pred,output_dict=True)
        st.dataframe(classif,width=1000)
        #st.info(f'classification report is : {classif}')

        st.subheader('CONFUSION MATRIX')
        # creating confusion matrix from trained model
        from sklearn.metrics import confusion_matrix
        dt_model = confusion_matrix(y_actual, y_pred)
        #st.error(dt_model)  # displaying confusion matrix

        # displaying confusion matrix as heatmap

        sns.heatmap(dt_model,color='k')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


if model_selection == 'RANDOM FOREST':
    st.title("INTEGRATING GUI with ML MODELS")  # setting title of webpage

    # taking new user inputs
    warehouse = st.number_input('ENTER WAREHOUSE BLOCK', step=1, min_value=1, max_value=6)
    customer_care_calls = st.number_input('ENTER CUSTOMER CARE CALLS', step=1)
    customer_rating = st.number_input('ENTER CUSTOMER RATING', step=1, min_value=1, max_value=5)
    cost_of_the_product = st.number_input('ENTER COST OF THE PRODUCT', step=1)
    prior_purchases = st.number_input('ENTER PRIOR PURCHASES', step=1)
    product_importance = st.number_input('ENTER PRODUCT IMPORTANCE', step=1, min_value=1, max_value=3)
    gender = st.number_input('ENTER GENDER', step=1, min_value=1, max_value=2)
    discount_offered = st.number_input('ENTER DISCOUNT OFFERED', step=1)
    weight_in_gms = st.number_input('ENTER WEIGHT IN GRAMS', step=1)
    reached_on_Time_YN = st.number_input('ENTER REACHED ON TIME', step=1, min_value=0, max_value=1)

    # storing new user input into a 2d array
    new_user_input = [
        [warehouse, customer_care_calls, customer_rating, cost_of_the_product, prior_purchases, product_importance,
         gender, discount_offered, weight_in_gms, reached_on_Time_YN]]

    from sklearn.ensemble import RandomForestClassifier

    # segregating data into features and target
    x_columns = ['Warehouse_block', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
                 'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms',
                 'Reached.on.Time_Y.N']  # definig input columns
    x = data_pda[x_columns].values  # choosing input columns

    y = data_pda['Mode_of_Shipment'].values  # choosing output column

    # splitting data into training and testing partitions
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2000, test_size=0.11)

    # 3. MODEL TRAINING
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

    submit_button = st.button("SUBMIT")
    if submit_button:
        # making the model predict answeers for x_test
        y_pred = rf_model.predict(x_test)

        # defining actual answers
        y_actual = y_test
        # making
        st.subheader('MODEL DIAGNOSIS')
        rf_model_output = rf_model.predict(new_user_input)
        if rf_model_output[0] == 1:
            st.error('THE SHIPMENT MODE IS FLIGHT')
        if rf_model_output[0] == 2:
            st.success('THE SHIPMENT MODE IS SHIP')
        if rf_model_output[0] == 3:
            st.success('THE SHIPMENT MODE IS ROAD')

        st.subheader('MODEL PARAMETER')

        # comparing actual answers with predictions from the model
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix,classification_report
        accuracy = accuracy_score(y_pred, y_actual)
        st.success(f'accuracy : {accuracy}')



        from sklearn.metrics import classification_report
        st.subheader('CLASSIFICATION REPORT')
        classif = classification_report(y_actual, y_pred, output_dict=True)
        st.dataframe(classif, width=1000)        #st.info(f'classification report is : {classif}' )

        # creating confusion matrix from trained model
        from sklearn.metrics import confusion_matrix
        st.subheader('CONFUSION MATRIX')
        rf_model = confusion_matrix(y_actual, y_pred)
      #  st.success(rf_model)  # displaying confusion matrix

        # displaying confusion matrix as heatmap
        import seaborn as sns

        sns.heatmap(rf_model)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


if model_selection == 'XGBOOST':
    st.title("INTEGRATING GUI with ML MODELS")  # setting title of webpage

    # taking new user inputs
    warehouse = st.number_input('ENTER WAREHOUSE BLOCK', step=1, min_value=1, max_value=6)
    customer_care_calls = st.number_input('ENTER CUSTOMER CARE CALLS', step=1)
    customer_rating = st.number_input('ENTER CUSTOMER RATING', step=1, min_value=1, max_value=5)
    cost_of_the_product = st.number_input('ENTER COST OF THE PRODUCT', step=1)
    prior_purchases = st.number_input('ENTER PRIOR PURCHASES', step=1)
    product_importance = st.number_input('ENTER PRODUCT IMPORTANCE', step=1, min_value=1, max_value=3)
    gender = st.number_input('ENTER GENDER', step=1, min_value=1, max_value=2)
    discount_offered = st.number_input('ENTER DISCOUNT OFFERED', step=1)
    weight_in_gms = st.number_input('ENTER WEIGHT IN GRAMS', step=1)
    reached_on_Time_YN = st.number_input('ENTER REACHED ON TIME', step=1, min_value=0, max_value=1)

    # storing new user input into a 2d array
    new_user_input = [
        [warehouse, customer_care_calls, customer_rating, cost_of_the_product, prior_purchases, product_importance,
         gender, discount_offered, weight_in_gms, reached_on_Time_YN]]

    from xgboost import XGBClassifier
    data_pda['Mode_of_Shipment'] = data_pda['Mode_of_Shipment'].map({1: 0, 2: 1, 3: 2})

    # segregating data into features and target
    x_columns = ['Warehouse_block', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
                 'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms',
                 'Reached.on.Time_Y.N']  # definig input columns
    x = data_pda[x_columns].values  # choosing input columns

    y = data_pda['Mode_of_Shipment'].values  # choosing output column

    # splitting data into training and testing partitions
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2000, test_size=0.1)

    # 3. MODEL TRAINING
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    submit_button = st.button("SUBMIT")
    if submit_button:
        # making the model predict answeers for x_test
        y_pred = xgb.predict(x_test)

        # defining actual answers
        y_actual = y_test

        # making
        st.subheader('MODEL DIAGNOSIS')
        xgb_model_output = xgb.predict(new_user_input)
        if xgb_model_output[0] == 1:
            st.error('THE SHIPMENT MODE IS FLIGHT')
        if xgb_model_output[0] == 2:
            st.success('THE SHIPMENT MODE IS SHIP')
        if xgb_model_output[0] == 3:
            st.success('THE SHIPMENT MODE IS ROAD')

        st.subheader('MODEL PARAMETER')
        # comparing actual answers with predictions from the model
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix,classification_report
        accuracy = accuracy_score(y_pred, y_actual)
        st.success(f'accuracy : {accuracy}')


        from sklearn.metrics import classification_report
        st.subheader('CLASSIFICATION REPORT')
        classif = classification_report(y_actual, y_pred, output_dict=True)
        st.dataframe(classif, width=1000)        #st.info(f'classification report is : {classif}')

        # creating confusion matrix from trained model
        from sklearn.metrics import confusion_matrix
        st.subheader('CONFUSION MATRIX')
        xgb_model = confusion_matrix(y_actual, y_pred)
        #st.success(xgb_model)  # displaying confusion matrix

        # displaying confusion matrix as heatmap
        import seaborn as sns

        sns.heatmap(xgb_model)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

if model_selection == 'MODEL PARAMETERS':
    st.subheader('ACCURACY')
    # accuracy
    st.set_option('deprecation.showPyplotGlobalUse', False)

    labels = ['DECISION TREE', "RANDOM FOREST", "XGBOOST"]
    accuracy = [53.09, 70.66, 68.91]
    import seaborn as sns
    sns.barplot(x=labels, y=accuracy)
    st.pyplot()

    st.subheader('PRECISION')
    # precison
    st.set_option('deprecation.showPyplotGlobalUse', False)

    labels = ['DECISION TREE', "RANDOM FOREST", "XGBOOST"]
    precision = [71.55, 71.5, 71.16]
    import seaborn as sns
    sns.barplot(x=labels, y=precision)
    st.pyplot()

    st.subheader('RECALL SCORE')
    # accuracy
    st.set_option('deprecation.showPyplotGlobalUse', False)

    labels = ['DECISION TREE', "RANDOM FOREST", "XGBOOST"]
    recall = [67.92, 98.84, 96.3]
    import seaborn as sns

    sns.barplot(x=labels, y=recall)
    st.pyplot()

    st.subheader('F1 SCORE')
    # accuracy
    st.set_option('deprecation.showPyplotGlobalUse', False)

    labels = ['DECISION TREE', "RANDOM FOREST", "XGBOOST"]
    f1 = [69.69, 82.98, 81.84]
    import seaborn as sns

    sns.barplot(x=labels, y=f1)
    st.pyplot()