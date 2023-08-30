###########      IMPORT LIBRAIRIES      ############

import shap, matplotlib
import streamlit.components.v1 as components
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='white')
matplotlib.use('Agg')

def main():
    
    st.set_page_config(layout="wide")


############      LOADING DATASET      ############


    df = pd.read_csv('diabetes_2.csv')


############      REPLACING MISSING VALUES    ############


    # Some features in the dataset have a value of 0, which denotes missing data. --> replace 0 for NaN
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    # glucose, blood pressure, skin thickness, insulin, and BMI, all have missing values. Use the ‘Outcome’ variable to find the mean to replace missing data

    @st.cache(persist= True)
    def median_target(var):   
            temp = df[df[var].notnull()]
            temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
            return temp
        
    # Function to find the mean to replace missing data

        #Glucose
    median_target("Glucose")
    df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
    df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3
    
        #Blood Pressure
    median_target("BloodPressure")
    df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
    df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3
    
        #Skin Thickness
    median_target("SkinThickness")
    df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
    df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0
    
        #Insulin
    median_target("Insulin")
    df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
    df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8
    
        #Bmi
    median_target("BMI")
    df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
    df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4
    

    #Function to plot shapley values tightly
    shap.initjs() 
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height) 
    
    
############      MODEL BUILDING      ############


     # Function to Split the data into independent 'X' and dependent 'y' variables (predictor and target variables)
    @st.cache(persist=True)
    def split(df):
        X = df.drop(columns='Outcome')
        y = df['Outcome']
     
         # On met tout sur la même échelle
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
         #X = pd.get_dummies(df)
         # Split the dataset into 80% Training set and 20% Testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        return X_train, X_test, y_train, y_test


    X_train, X_test, y_train, y_test = split(df)


     #create and train the model
    GB = GradientBoostingClassifier()
    GB.fit(X_train, y_train)
    

    #Open and display logo image
    st.subheader('DIABETES DETECTION')
    image = Image.open('logo_business_et_decision.jpg')
    st.sidebar.image(image, caption='', width=250)


########     ADD SUBMIT BUTTON FORM      #########


    with st.sidebar.form(key='Medical Data'):
        st.text_area("PLEASE INSERT PATIENT'S INFORMATION")
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=None, value=0, step=1)
        glucose = st.number_input('Glucose', min_value=0.00, max_value=None, value=121.69, step=0.01)
        blood_pressure = st.number_input('BloodPressure', min_value=0.00, max_value=None, value=72.42, step=0.01) 
        skin_thickness = st.number_input('SkinThickness', min_value=0.00, max_value=None, value=29.24, step=0.01)
        insulin = st.number_input('Insulin', min_value=0.00, max_value=None, value=156.99, step=0.01)
        BMI = st.number_input('BMI', min_value=0.00, max_value=None, value=32.44, step=0.01)
        DPF = st.number_input('DiabetesPedigreeFunction', min_value=0.00, max_value=None, value=0.47, step=0.01)
        age = st.number_input('Age', min_value=21, max_value=None, value=33, step=1)
        submit_button = st.form_submit_button(label='Submit Inputs')
        
    #Function for the input from the user
    def get_user_input():
        #store a dictionary into a variable
        user_data = {'Pregnancies': pregnancies,
                     'Glucose': glucose,
                     'BloodPressure': blood_pressure,
                     'SkinThickness': skin_thickness,
                     'Insulin': insulin,
                     'BMI': BMI,
                     'DiabetesPedigreeFunction': DPF,
                     'Age': age
                     }
        #transform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features
            
    #store the user input into a variable  
    user_input = get_user_input()
    

#######     MODEL PREDICTIONS & PROBAS      #######


    prediction = GB.predict(user_input) # Store the user's predictions in a variable
    proba_prediction = GB.predict_proba(user_input) # Store the user's probas predictions ina variable
    y_test_pred = GB.predict(X_test) # Store the model's predictions in a variable


#########        SHAPLEY GRAPH         ##########

    scaler = StandardScaler()
    X = df.drop(columns='Outcome')
    X = pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = df['Outcome']
    from sklearn.ensemble import RandomForestClassifier
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    #gb_model = GradientBoostingClassifier(random_state=0).fit(train_X, train_y)
    gb_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
    
    explainer = shap.TreeExplainer(gb_model) #explain model's predictions
    shap_values = explainer.shap_values(user_input)
    
    
########        Feature Importances       ########

    #create a bar chart for feature importances
    #Create a function to plot feature importances
    @st.cache(persist= True)
    def plot_feature_importance(importance,names,model_type):

        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        #Create a DataFrame using a Dictionary
        data={'Names':feature_names,'Importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['Importance'], ascending=False,inplace=True)
        #Define size of bar plot
            #plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['Importance'], y=fi_df['Names'])
        #Add chart labels
        plt.title(model_type + ' FEATURE IMPORTANCE')
            #plt.xlabel('FEATURE IMPORTANCE')
            #plt.ylabel('FEATURE NAMES')


########      LAYOUT Application       ########


    st.set_option('deprecation.showPyplotGlobalUse', False)
    container1 = st.container()
    col1, col2 = st.columns(2)
    with container1:
        with col1:
            #set a subheader and display the class probability
            st.write("Class PROBABILITY in %")
            st.metric(label='Diabetic', value=((proba_prediction[0,1])*100).round(2))
            st.metric(label='No Diabetic', value=((proba_prediction[0,0])*100).round(2))
        
        with col2:
        # We can also just take the mean absolute value of the SHAP values for each feature 
        # to get a standard bar plot 
            plt.title('Assessing prediction feature importances based on Shap values')
            #shap.summary_plot(shap_values, user_input.values, plot_type="bar", feature_names = user_input.columns)
            #shap.summary_plot(shap_values, user_input)
            class_names = ['Diabetic','No Diabetic']
            shap.summary_plot(shap_values, user_input, class_names = class_names)
            st.pyplot(bbox_inches='tight')
            plt.clf()
        
    container2 = st.container()
    col3, col4 = st.columns(2)
    with container2:
        with col3:
            fig9 = plot_feature_importance(GB.feature_importances_,X_train.columns,'GB Classifier ')
            st.pyplot(fig9, bbox_inches='tight')
            plt.clf()
        
        with col4:
            # Plotting the KDE Plot with "check" form
            with st.form(key='Kernel Density'):
                option = st.selectbox('VARIABLE SELECTION',
                ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
                
                if option == 'Pregnancies':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'Pregnancies'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'Pregnancies'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['pregnancies'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('Pregnancies')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                
                if option == 'Glucose':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'Glucose'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'Glucose'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['glucose'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('Glucose')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
            
                if option == 'BloodPressure':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'BloodPressure'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'BloodPressure'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['blood_pressure'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('BloodPressure')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                    
                if option == 'SkinThickness':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'SkinThickness'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'SkinThickness'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['skin_thickness'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('SkinThickness')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                        
                if option == 'Insulin':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'Insulin'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'Insulin'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['insulin'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('Insulin')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                            
                if option == 'BMI':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'BMI'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'BMI'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['BMI'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('BMI')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                                
                if option == 'DiabetesPedigreeFunction':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'DiabetesPedigreeFunction'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'DiabetesPedigreeFunction'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['DPF'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('D.Pedigree F°')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                                    
                if option == 'Age':
                    sns.kdeplot(df.loc[(df['Outcome']==0),'Age'], vertical=False, color='g', shade=True, label='No Diabetic')
                    sns.kdeplot(df.loc[(df['Outcome']==1),'Age'], vertical=False, color='r', shade=True, label='Diabetic')
                        #sns.stripplot(data = user_input['age'], orient = 'h', color = "black", label='User Input')
                    # Setting the X and Y Label
                    plt.xlabel('Age')
                    plt.ylabel('Probability Density')
                    plt.legend()
                        #plt.tight_layout()
                    st.pyplot(bbox_inches='tight')
                    plt.clf()
                submit_button_2 = st.form_submit_button(label='Select Variable')

    option2 = st.selectbox('ACCESSING PERFORMANCE',('None', 'Display'))
    
    if option2 == None:
        st.selectbox('ACCESSING PERFORMANCE',('None'))
    
    if option2 == 'Display':
        container3 = st.container()
        col5, col6 = st.columns(2)
        with container3:
            with col5:
                image = Image.open('boxplot.png')
                st.image(image, caption='Read a Boxplot')
                with st.form(key='Boxplots'):
                    option3 = st.selectbox('VARIABLE SELECTION',
                    ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age'))
                    
                    if option3 == 'Pregnancies':
                    ###   PREGNANCIES   ###
                        fig1, ax1 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['Pregnancies'], color="plum", width=.5)
                        sns.stripplot(data = user_input['Pregnancies'], orient = 'h', color = "purple")
                        plt.title('PREGNANCIES Distribution', fontsize=14)
                        plt.xlabel('Values')
                        ax1.set(xlim=(0,18))
                        props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
                        ax1.text(15, 0.4, "Purple Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig1)
                        plt.clf()
                    
                    if option3 == 'Glucose':
                    ###   GLUCOSE   ###
                        fig2, ax2 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['Glucose'], color="red", width=.5)
                        sns.stripplot(data = user_input['Glucose'], orient = 'h', color = "black")
                        plt.title('GLUCOSE Distribution', fontsize=14)
                        plt.xlabel('Values in mg/dL')
                        ax2.set(xlim=(0,200))
                        props = dict(boxstyle='round', facecolor='red', alpha=0.2)
                        ax2.text(168, 0.4, "Black Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.clf()
                    
                    if option3 == 'BloodPressure':
                    ###   BLOOD PRESSURE   ###
                        fig3, ax3 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['BloodPressure'], color="grey", width=.5)
                        sns.stripplot(data = user_input['BloodPressure'], orient = 'h', color = "red")
                        plt.title('BLOOD PRESSURE Distribution', fontsize=14)
                        plt.xlabel('Values in mmHg')
                        ax3.set(xlim=(0,250))
                        props = dict(boxstyle='round', facecolor='black', alpha=0.2)
                        ax3.text(212, 0.4, "Red Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.clf()
                    
                    if option3 == 'SkinThickness':
                    ###   SKIN THICKNESS   ###
                        fig4, ax4 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['SkinThickness'], color="plum", width=.5)
                        sns.stripplot(data = user_input['SkinThickness'], orient = 'h', color = "purple")
                        plt.title('SKIN THICKNESS Distribution', fontsize=14)
                        plt.xlabel('Values in mm')
                        ax4.set(xlim=(0,100))
                        props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
                        ax4.text(84, 0.4, "Purple Dot  User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig4)
                        plt.clf()

                    if option3 == 'Insulin':
                    ###   INSULIN   ###
                        fig5, ax5 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['Insulin'], color="red", width=.5)
                        sns.stripplot(data = user_input['Insulin'], orient = 'h', color = "black")
                        plt.title('INSULIN Distribution', fontsize=14)
                        plt.xlabel('Values in mlU/L')
                        ax5.set(xlim=(0,999))
                        props = dict(boxstyle='round', facecolor='red', alpha=0.2)
                        ax5.text(835, 0.4, "Black Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig5)
                        plt.clf()

                    if option3 == 'BMI':
                    ###   BMI   ###
                        fig6, ax6 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['BMI'], color="grey", width=.5)
                        sns.stripplot(data = user_input['BMI'], orient = 'h', color = "red")
                        plt.title('BODY MASS INDEX Distribution', fontsize=14)
                        plt.xlabel('Values')
                        ax6.set(xlim=(0,70))
                        props = dict(boxstyle='round', facecolor='black', alpha=0.2)
                        ax6.text(59, 0.4, "Red Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig6)
                        plt.clf()
                        
                    if option3 == 'DPF':
                    ###   DPF   ###
                        fig7, ax7 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['DiabetesPedigreeFunction'], color="plum", width=.5)
                        sns.stripplot(data = user_input['DiabetesPedigreeFunction'], orient = 'h', color = "purple")
                        plt.title('DIABETES PEDIGREE F° Distribution', fontsize=14)
                        plt.xlabel('Values in %')
                        ax7.set(xlim=(0,2.500))
                        props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
                        ax7.text(2.08, 0.4, "Purple Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig7)
                        plt.clf()
                    
                    if option3 == 'Age':
                    ###   AGE   ###
                        fig8, ax8 = plt.subplots(figsize=(10,2))
                        sns.boxplot(df['Age'], color="red", width=.5)
                        sns.stripplot(data = user_input['Age'], orient = 'h', color = "black")
                        plt.title('AGE Distribution', fontsize=14)
                        plt.xlabel('Values')
                        ax8.set(xlim=(0,110))
                        props = dict(boxstyle='round', facecolor='red', alpha=0.2)
                        ax8.text(92, 0.4, "Black Dot = User Input", fontsize=10, bbox=props)
                        plt.grid()
                        plt.tight_layout()
                        st.pyplot(fig8)
                        plt.clf()
                    submit_button_3 = st.form_submit_button(label='Select Variable')

            with col6:
                with st.form(key='CF&ROC'):
                    option4 = st.selectbox('PERFORMANCE SELECTION',
                    ('Confusion Matrix', 'ROC Curve'))

                    if option4 == 'Confusion Matrix':
                        st.write('CONFUSION MATRIX')
                        
                        def confusion_matrix_plot (y_test, prediction2):
                            cm = confusion_matrix(y_test, prediction2)
                            classes = ['0', '1']
                            figure, ax = plot_confusion_matrix(conf_mat = cm,
                                                               class_names = classes,
                                                               show_absolute = True,
                                                               show_normed = False,
                                                               colorbar = True)
            
                        prediction2 = GB.predict(X_test)

                        # Get and display sensitivity and specificity score
                        cm = confusion_matrix(y_test, prediction2)
                        sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
                        specificity = cm[0,0]/(cm[0,0]+cm[0,1])
                        
                        st.pyplot(confusion_matrix_plot(y_test, prediction2))
                        st.write('Sensitivity: ', str(sensitivity))
                        st.write('Specificity: ', str(specificity))
                        
                    if option4 == 'ROC Curve':

                        st.write('ROC CURVE')
        
                        # Instantiate the classfiers and make a list
                        classifier = [GB]
                        # Define a result table as a DataFrame
                        result_table = pd.DataFrame(columns=['classifier', 'fpr', 'tpr', 'auc'])
                        # Train the models and record the results
                        for cls in classifier:
                            model = cls.fit(X_train, y_train)
                            yproba = model.predict_proba(X_test)[:,1] # braquets here to say to keep probas for the positive outcome only
        
                            fpr, tpr, thresholds = roc_curve(y_test,  yproba)
                            J = tpr - fpr
                            ix = argmax(J)
                            best_thresh = thresholds[ix]
                            auc = roc_auc_score(y_test, yproba)
        
                            result_table = result_table.append({'classifier':cls.__class__.__name__,
                                                                'fpr':fpr, 
                                                                'tpr':tpr, 
                                                                'auc':auc}, ignore_index=True)                                    
            
                        # Set name of the classifiers as index labels
                        result_table.set_index('classifier', inplace=True)
                        figgg = plt.figure(figsize=(15,10))
                
                        for i in result_table.index:
                            plt.plot(result_table.loc[i]['fpr'], 
                                     result_table.loc[i]['tpr'], 
                                     label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
            
                        plt.plot([0,1], [0,1], color='orange', linestyle='--')
                        #plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best_threshold ')

                        plt.xticks(np.arange(0.0, 1.1, step=0.1))
                        plt.xlabel("Flase Positive Rate", fontsize=15)

                        plt.yticks(np.arange(0.0, 1.1, step=0.1))
                        plt.ylabel("True Positive Rate", fontsize=15)

                        plt.title('ROC Curves', fontweight='bold', fontsize=15)
                        plt.legend(prop={'size':13}, loc='lower right')
                        plt.grid()
                        plt.tight_layout()

                        st.pyplot(figgg)
                        st.write('Best Threshold = %f' % (best_thresh))
                    submit_button_4 = st.form_submit_button(label='Select Performance')

    
if __name__== '__main__':
    main()