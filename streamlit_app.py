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
  df = pd.read_csv("cleaned_breast_cancer_data.csv")
  df
