import pickle
import tensorflow
import numpy as np
import pandas as pd
import streamlit as st
from keras import models
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score



st.set_page_config(
    page_title='Adult income Model',
    page_icon='💸',
    layout='centered'
)



st.title('Welcome to Incoming Prediction App')

_, col_tab = st.columns([2, 20])
st.markdown('<br>', unsafe_allow_html=True)
model_radio = col_tab.radio('*Models*:', ['Logistic Regression', 'Multi Layer Preprocessing (MLP)'], index=0, key='Special_radio')
if model_radio == 'Logistic Regression':

    with open(r'lr_model.pkl', 'rb') as f:
        model =   pickle.load(f)
else:
    model =  models.load_model(r'mlp-model.h5')




class Preparation:
    def remove_nan_row(self, df:pd.DataFrame):
        df = df.replace('?', np.nan)
        df = df.dropna()
        return df
    
    def scale_numerical(self, df, numerical_columns):
        self.numeric_scaler = StandardScaler()
        df[numerical_columns] = self.numeric_scaler.fit_transform(df[numerical_columns])
        return df
    
    def scale_numerical_transform(self, df, numerical_columns):
        df[numerical_columns] = self.numeric_scaler.transform(df[numerical_columns])
        return df
    
    def label_encoding(self, df, le_cols:list):
        keys = {}
        for col in le_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            keys[col] = le.classes_
        return df, keys
    
    def label_encoding_transform(self, df, le_cols: list, keys: dict):
        for col in le_cols:
            df[col] = df[col].apply(lambda val: np.int64(list(keys[col]).index(val)))
        return df


    def expand_columns(self, df, columns):
        expand_data = pd.get_dummies(df, columns=columns, prefix=columns)
        self.dummies_columns = expand_data.iloc[:, 8:].columns
        return expand_data
    
    def expand_columns_transform(self, df, columns):
        contained_cols = pd.get_dummies(df, columns=columns, prefix=columns).iloc[:, 7:].columns # the dummies columns which are in df
        data = df.copy()
        for n_col in self.dummies_columns: # n_col -> new column
            if n_col in contained_cols:
                data.loc[0, n_col] = 1
            else:
                data.loc[0, n_col] = 0
        data = data.drop(columns=columns, axis=1)
        data[self.dummies_columns] = data[self.dummies_columns].astype(np.bool_)

        return data





    def reorder_columns(self, df, current_order):
        current_order.remove('income')
        current_order.append('income')
        df = df[current_order]
        return df
    
    def split_train_test(self, df, random_state=42, test_size=.2):
        '''the income (target column) is selected the -1 (last column) by default'''
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                    random_state=random_state,
                                                    test_size=test_size,
                                                    stratify=df.iloc[:, -1]

        )
        return x_train, x_test, y_train, y_test
    

    def evaluate(self, y_test, y_pred, metric:str='accuracy'):
        if metric == 'accuracy':
            return accuracy_score(y_test, y_pred)
    
   
    
        elif metric == 'loss':
            return zero_one_loss(y_test, y_pred)
        
        elif metric == 'conf_matrix':
            return confusion_matrix(y_test, y_pred)
        








df = pd.read_csv(r'adult-income.csv')
DataPreprocessor = Preparation()
df = DataPreprocessor.remove_nan_row(df)
df_copy = df.copy(True)

numerical_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
data = DataPreprocessor.scale_numerical(df_copy, numerical_columns)
data, le_keys = DataPreprocessor.label_encoding(data, ['gender', 'income'])
income_keys = list(le_keys['income'])

dummies_columns_name = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
expand_data = DataPreprocessor.expand_columns(data, dummies_columns_name)

expand_data = DataPreprocessor.reorder_columns(expand_data, list(expand_data.columns))
data = DataPreprocessor.reorder_columns(data, list(data.columns))


with st.expander('View DataFrame', False):
    rows = st.number_input('How many rows do you want to see?', 1, df.shape[0]+1, value=8)
    from_head = st.checkbox('Show from Top', True)
    st.dataframe(df.head(rows) if from_head else df.tail(rows))

    st.subheader('DataFrame After Preparations will be like this:')
    st.dataframe(expand_data.head(rows) if from_head else expand_data.tail(rows))


with st.expander('**Predict Data**', True):
    predict_radio = st.radio('Which method do you want to predict the data:', [
        'Use a specific row of dataframe',
        'Define a row'
    ], index=0)


    if predict_radio == 'Use a specific row of dataframe':
        index = st.number_input('which row to you want to use?', 0, expand_data.shape[0], value=0)
        st.markdown('<br>', unsafe_allow_html=True)
        row = expand_data.iloc[index:index+1, :-1]
        st.dataframe(row)
        predict_col, actual_col = st.columns(2)
        if model_radio ==  'Logistic Regression':
            predicted_class = model.predict(row)
            predicted_class_prob = round(float(model.predict_proba(row)[0][predicted_class]), 5) * 100
            predict_col.metric('Predicted Clasxs', predicted_class, delta=predicted_class_prob)
        else:
            predicted_class_prob = round(float(model.predict(row)[0][0]), 5)
            predicted_class = int(predicted_class_prob > 0.5)
            predict_col.metric('Predicted Class', predicted_class, delta=round(float(predicted_class_prob), 5)*100 if predicted_class_prob > 0.5 else 100 - round(float(predicted_class_prob), 5)*100)


        actual_col.metric('Actual class',expand_data.iloc[index, -1])


    elif predict_radio == 'Define a row':
        form = st.form('get')
        row = []

        age = form.number_input('Age', df['age'].min(), df['age'].max(), value=30)
        row.append(age)

        workclasses = list(df.workclass.value_counts().reset_index()['workclass'])
        workclass = form.selectbox('workclass', workclasses)
        row.append(workclass)

        fnlwgt = form.number_input('fnlwgt', df['fnlwgt'].min()-3000, df['fnlwgt'].max()+10000, value=200_000)
        row.append(fnlwgt)

        educations = list(df.education.value_counts().reset_index()['education'])
        education = form.selectbox('education', educations)
        row.append(education)
    
        education_num = form.number_input('educational-num', df['educational-num'].min(), df['educational-num'].max(), value=4)
        row.append(education_num)

        marital_statuses = list(df['marital-status'].value_counts().reset_index()['marital-status'])
        marital_status = form.selectbox('marital-status', marital_statuses)
        row.append(marital_status)

        occupations = list(df['occupation'].value_counts().reset_index()['occupation'])
        occupation = form.selectbox('occupation', occupations)
        row.append(occupation)

        relationships = list(df['relationship'].value_counts().reset_index()['relationship'])
        relationship = form.selectbox('relationship', relationships)
        row.append(relationship)

        races = list(df['race'].value_counts().reset_index()['race'])
        race = form.selectbox('race', races)
        row.append(race)

        genders = list(df['gender'].value_counts().reset_index()['gender'])
        gender = form.selectbox('gender', genders)
        row.append(gender)


        capital_gain = form.number_input('capital-gain', df['capital-gain'].min(), df['capital-gain'].max(), value=10)
        row.append(capital_gain)

        capital_loss = form.number_input('capital-loss', df['capital-loss'].min(), df['capital-loss'].max(), value=15)
        row.append(capital_loss)

        hours = form.number_input('hours-per-week', df['hours-per-week'].min(), df['hours-per-week'].max(), value=30)
        row.append(hours)



        native_countries = list(df['native-country'].value_counts().reset_index()['native-country'])
        native_county = form.selectbox('native-country', native_countries)
        row.append(native_county)


        row = np.array(row)
        row = row.reshape(1, -1)
        row_copy = row.copy()


        row_df = pd.DataFrame(row_copy, columns=df.columns[:-1])

        row_scaled = DataPreprocessor.scale_numerical_transform(row_df, numerical_columns)
        row_label = DataPreprocessor.label_encoding_transform(row_scaled, ['gender'], le_keys)

        row_expand = DataPreprocessor.expand_columns_transform(row_label, dummies_columns_name)


        submit = form.form_submit_button()


        if True:
            if model_radio == 'Logistic Regression':
                predicted_class = int(model.predict(row_expand))
                predicted_class_prob = round(float(model.predict_proba(row_expand)[0][predicted_class]), 5)*100
                predicted_value = income_keys[predicted_class]
                st.metric('Predicted Class', predicted_value, delta=predicted_class_prob)
                
            else:
                predicted_class_prob = round(float(model.predict(row_expand)[0][0]), 5)
                predicted_class = int(predicted_class_prob > 0.5)
                predicted_value = income_keys[predicted_class]
                st.metric('Predicted Class', predicted_value, delta=predicted_class_prob if predicted_class_prob > 0.5 else 1 - predicted_class_prob)


with st.expander('*Model Evaluations*', False):
    def calc_predict_mlp():    
        predicted_classes_prob = model.predict(expand_data.iloc[:, :-1])
        predicted_classes = predicted_classes_prob > 0.5
        predicted_classes = predicted_classes.reshape(predicted_classes.shape[0])
        actual_classes = expand_data.iloc[:, -1]
        return predicted_classes, actual_classes
    
    def calc_predict_lr():    
        predicted_classes_prob = model.predict(expand_data.iloc[:, :-1])
        predicted_classes = predicted_classes_prob > 0.5
        predicted_classes = predicted_classes.reshape(predicted_classes.shape[0])
        actual_classes = expand_data.iloc[:, -1]
        return predicted_classes, actual_classes
    
    if model_radio == 'Logistic Regression':
        predicted_classes, actual_classes = calc_predict_lr()
    else:
        predicted_classes, actual_classes = calc_predict_mlp()
    
    accuracy = DataPreprocessor.evaluate(actual_classes, predicted_classes, 'accuracy')
    loss = DataPreprocessor.evaluate(actual_classes, predicted_classes, 'loss')
    conf_matrix = DataPreprocessor.evaluate(actual_classes, predicted_classes, 'conf_matrix')

    accuracy_col, loss_col, conf_col = st.columns([1, 1, 3])

    accuracy_col.metric("Accuracy", round(accuracy, 5)*100)
    loss_col.metric('Loss', round(loss, 5)*100)
    conf_col.write('Confusion matrix')
    conf_matrix = pd.DataFrame(conf_matrix, columns=['predict +', ' predict -'], index=['actual +', 'actual-'])
    conf_col.table(conf_matrix)