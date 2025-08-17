import streamlit as st
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler



st.title('🎈ML CANCER DIAGNOSIS APP')

st.info('MACHINE LEARNING PIPELINE')
with st.expander("Data"):
  st.write("**Raw Data**")
  df= pd.read_csv("cleaned_breast_cancer_data.csv")
  df['diagnosis'].replace({'M':1,'B':0},inplace=True)
  df

# Input features
with st.sidebar:
  st.header('Input features')
  
  texture_worst = st.slider('texture worse', -2.7, 2.7	, -0.0)
  texture_se = st.slider('texture_se', -2.6, 2.6, -0.0)
  texture_mean = st.slider('texture_mean', -6.8	, 6.0, -0.5)
  symmetry_worst= st.slider('symmetry_worst', -2.7, -2.7, 0.0) 
  symmetry_se= st.slider('symmetry_se', -6.0, 6.3, -0.0)
  symmetry_mean= st.slider('symmetry_mean', -7.2, 6.4, -0.4)
  smoothness_worst= st.slider('smoothness_worst', -11.9	, 11.7, -0.1)
  radius_worst= st.slider('radius_worst', -2.7, 2.7, -0.7)
  smoothness_se= st.slider('smoothness_se', -4.6, 6.6, 1.0)
  smoothness_mean= st.slider('smoothness_mean', -13.2, 10.3, -1.5)

   
  # Create a DataFrame for the input features
  data = {'texture_worst': texture_worst,
          'texture_se': texture_se,
          'texture_mean': texture_mean,
          'symmetry_worst': symmetry_worst,
          'symmetry_se': symmetry_se,
          'symmetry_mean': symmetry_mean,
          'smoothness_worst': smoothness_worst,
          'radius_worst': radius_worst,
          'smoothness_se': smoothness_se,
          'smoothness_mean': smoothness_mean}
  input_df = pd.DataFrame(data, index=[0])
  input_data = pd.concat([input_df], axis=0)



    
    
  

