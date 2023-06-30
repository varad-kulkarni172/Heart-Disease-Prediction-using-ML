# heart disease prediction model using random forest classifier
# Import libraries
# Pandas used to read the dataset
import pandas as pd
# Numpy is used for arrays and matrices
import numpy as np
# Matplotlib used to draw graphs or diagrams(ROC Curve)
import matplotlib.pyplot as plt
# Sklearn used to train the model and calculate accuracy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
# Load the data
df = pd.read_csv('heart.csv')
# Display the first five rows of the data
df.head()
# Check the shape of the data
df.shape
# Check the data types
print(df.dtypes)
# print(df)
# Check for missing values
df.isnull().sum()
# Check the class distribution
df['target'].value_counts()
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.345, random_state=42)

# Check the shape of training set and test set
print(X_train.shape, X_test.shape)
# Create a random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
# Train the model
rf_clf.fit(X_train, y_train)
print("\n")
# Function to print the model's performance

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
       # print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
       # print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
# Print the model's performance
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
# Plot the ROC curve
y_pred_prob = rf_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.show()
# Print the AUC score
y_pred_prob = rf_clf.predict_proba(X_test)[:,1]
auc = metrics.roc_auc_score(y_test, y_pred_prob)
print("AUC Score of the model is ")
print(auc)
# Function to get the user's input
def get_input():
  age = int(input('Enter your age: '))
  gender = int(input('Enter your gender (0 for Male/1 for Female): '))
  cp = int(input('Enter your chest pain type (0-3): '))
  trestbps = int(input('Enter your resting blood pressure(mm/Hg): '))
  chol = int(input('Enter your cholesterol(mg/dl): '))
  fbs = int(input('Enter your fasting blood sugar (0 for <120 mg/dL/1 for >120 mg/dL): '))
  restecg = int(input('Enter your resting electrocardiographic results (0-2): '))
  thalach = int(input('Enter your maximum heart rate achieved: '))
  exang = int(input('Enter your exercise induced angina (0/1): '))
  oldpeak = float(input('Enter your ST depression induced by exercise relative to rest: '))
  slope = int(input('Enter the slope of the peak exercise ST segment (0-2): '))
  ca = int(input('Enter the number of major vessels (0-3) colored by fluoroscopy: '))
  thal = input('Enter your thalassemia (normal(1)/fixed defect(2)/reversable defect(3)): ')

  # Map the input to the appropriate format
  data = {
    'age': [age],
    'sex': [gender],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
  }
  input_df = pd.DataFrame(data)
  
  # Preprocess the data (if necessary)
  # (This may include scaling, encoding, etc.)
  
  return input_df

# Get the user's input
input_df = get_input()

# Use the model to make a prediction
prediction = rf_clf.predict(input_df)[0]
print('\n================================================================')
print('\n')

# Display the prediction
if prediction == 0:
  print('No need to worry! You dont have any heart disease.')
  print('Have a good day!')
else:
  print('We are sorry to inform you that you are predicted to have a heart disease.')
  print('Please consult a specialist as soon as possible.')
  print('Following are some of the specialists we recommend in some major cities :')
  print('\n======================== MAHARASHTRA ===========================')
  print('1.Ruby Hall Clinic - Pune')
  print('2.Kokilaben Dhirubhai Ambani Hospital - Mumbai')
  print('\n======================== UTTAR PRADESH ===========================')
  print('1.Awadh Hospital and Heart Centre - Lucknow')
  print('2.Heartline Cardiac Care Centre - Prayagraj')
  print('\n======================== TAMIL NADU ===========================')
  print('1.Apollo Hospital - Chennai')
  print('2.Kauvery Hospital - Tiruchirappalli')
  print('\n======================== GUJARAT ===========================')
  print('1.Apex Heart Institute - Ahmedabad')
  print('2.Nirmal Hospital Pvt Ltd - Surat')
  print('\n======================== PUNJAB ===========================')
  print('1.Fortis Hospital - Chandigarh')
  print('\n======================== DELHI ===========================')
  print('1.Apollo Hospital - Delhi')
  print('2.Fortis Escort Hospital - Delhi')
  print('\n======================== KARNATAKA ===========================')
  print('1.Manipal Hospital - Bengaluru')
  print('\n======================== RAJASTHAN ===========================')
  print('1.Global Heart and General Hospital - Jaipur')
  print('2.Verma Hospital - Jaisalmer')
  print('\n======================== WEST BENGAL ===========================')
  print('1.Calcutta Heart Clinic and Hospital - Kolkata')
  print('\n======================== ASSAM ===========================')
  print('1.Apollo Excelcare Hospital - Guwahati')
