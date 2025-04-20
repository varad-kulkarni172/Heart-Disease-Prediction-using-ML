import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from geopy.geocoders import Nominatim
import requests
import webbrowser

# Load and prepare the dataset
df = pd.read_csv('./content/heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.345, random_state=42)

# Train the RandomForest Classifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rf_clf.fit(X_train, y_train)

# Function to print model evaluation metrics
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)} \n")

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# Plot ROC Curve
y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.show()

auc = roc_auc_score(y_test, y_pred_prob)
print("AUC Score of the model is ")
print(auc)

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
    thal = int(input('Enter your thalassemia (normal(1)/fixed defect(2)/reversable defect(3)): '))

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

    return input_df

def find_nearest_hospital(location):
    # Geocode user's location using Nominatim
    geolocator = Nominatim(user_agent="hospital_locator")
    location = geolocator.geocode(location)
    lat, lon = location.latitude, location.longitude

    # Query for nearby hospitals using Overpass API
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node
      [amenity=hospital]
      (around:5000,{lat},{lon});
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Find the nearest hospital with a phone number
    nearest_hospital = None
    for element in data['elements']:
        name = element['tags'].get('name', 'N/A')
        address = element['tags'].get('addr:full', 'N/A')
        phone = element['tags'].get('phone', 'N/A')
        if phone != 'N/A':
            nearest_hospital = {'name': name, 'address': address, 'phone': phone}
            break

    return nearest_hospital

# Collect user input
input_df = get_input()

# Predict heart disease
prediction = rf_clf.predict(input_df)[0]
print('\n================================================================')
print('\n')

if prediction == 0:
    print('No need to worry! You don\'t have any heart disease.')
    print('Have a good day!')
else:
    print('We are sorry to inform you that you are predicted to have a heart disease.')
    print('Please consult a specialist as soon as possible.')
    print('Initiating precautions and remedies...')

    # Display precautions and remedies
    print("""
    ***Precautions and Remedies for Suspected Heart Disease***
    
    If you or someone you know is predicted with heart disease, it's crucial to take immediate precautions and initiate remedies while waiting to seek medical help. Here are steps to consider:
    
    --> Precautions:
    1. Immediate Rest: Sit or lie down comfortably in a relaxed position to reduce strain on the heart.
       
    2. Call for Assistance: Contact emergency services or someone nearby for immediate help.
    
    3. Avoid Physical Exertion: Refrain from engaging in strenuous activities or lifting heavy objects.
    
    4. Stay Calm: Anxiety can worsen symptoms, so try to remain calm and reassure the individual.
    
    --> Remedies:
    1. Take Aspirin: If not allergic and advised by a healthcare professional, taking a low-dose aspirin (81 mg) can help reduce blood clotting.
    
    2. Nitroglycerin: If prescribed previously and available, use nitroglycerin as directed for chest pain or discomfort.
    
    3. Keep Medications Handy: Have prescribed heart medications readily accessible for immediate use.
    
    4. Monitor Vital Signs: Check pulse rate and breathing regularly while awaiting medical assistance.
    
    --> Additional Care:
    1. Positioning: Prop up the person's head and shoulders with pillows to ease breathing.
      
    2. Loosen Clothing: Ensure clothing around the neck and chest is loose to aid in breathing.
    
    3. Stay Warm: Keep the person warm with blankets or clothing, as cold temperatures can stress the heart.
    
    --> Important Notes:
    1. Do Not Delay: If symptoms are severe or worsen, do not hesitate to call emergency services immediately.
    
    2. Communicate Clearly: Provide clear and concise information about symptoms and any relevant medical history to healthcare providers.
    
    3. Stay with the Person: If possible, stay with the individual until medical help arrives to provide reassurance and monitor their condition.
    
    Taking these precautions and initiating these remedies can potentially improve outcomes while waiting for professional medical assistance. Always follow the advice of healthcare professionals and seek emergency care promptly in case of suspected heart disease or related symptoms.
    """)

    # Find nearest hospital and initiate call
    location = input("Enter your location (e.g., city name): ")
    nearest_hospital = find_nearest_hospital(location)

    if nearest_hospital:
        print(f"Nearest Hospital: {nearest_hospital['name']}")
        print(f"Address: {nearest_hospital['address']}")
        print(f"Phone: {nearest_hospital['phone']}")

        # Provide instructions to the user for initiating the call
        print(f"Please use your phone to call {nearest_hospital['name']} at {nearest_hospital['phone']}")
    else:
        print("No hospitals found with available phone numbers nearby. Please contact emergency services.")
