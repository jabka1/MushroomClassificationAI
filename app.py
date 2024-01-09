from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DEFAULT_VALUE = 0

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/mushrooms.csv')

# Separate features and labels
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical columns separately
categorical_features = X.select_dtypes(include=['object']).columns
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])
    label_encoders[feature] = le

# Encode the target variable 'class'
label_encoder_class = LabelEncoder()
y = label_encoder_class.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict function
def predict_mushroom(params):
    input_data = []
    for feature, value in params.items():
        le = label_encoders.get(feature)
        if le:
            try:
                input_data.append(le.transform([value])[0])
            except ValueError as e:
                print(f"Warning: Unseen label in feature {feature}. {str(e)}")
                input_data.append(DEFAULT_VALUE)

    if len(input_data) == len(X.columns):
        try:
            prediction = model.predict([input_data])[0]
        except ValueError as e:
            print(f"Warning: Unseen label in target variable. {str(e)}")
            return 'Unknown'

        return 'Poisonous' if prediction == 1 else 'Edible' if prediction == 0 else 'Unknown'
    else:
        return 'Unknown'

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {}
        for key in df.columns[1:]:
            user_input[key] = request.form[key]
        prediction = predict_mushroom(user_input)
        return render_template('index.html', attributes=df.columns[1:], prediction=prediction)
    return render_template('index.html', attributes=df.columns[1:])

if __name__ == '__main__':
    app.run(debug=True)
