import numpy as np
import xgboost as xgb
import pickle

# Load the trained model
with open('xgboost_MFIloan_default_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Label Encoders for categorical features
with open('MFIlabel_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

def get_user_input():
    """Collects user input for loan features."""
    age = int(input("Enter age: "))
    income = float(input("Enter income: "))
    loan_amount = float(input("Enter loan amount: "))
    interest_rate = float(input("Enter interest rate: "))
    education = input("Enter education level (High School, Bachelor's, Master's): ")
    loan_term = int(input("Enter loan term (in months): "))
    months_employed = int(input("Enter months employed: "))
    employment_type = input("Enter employment type (Full-time, Part-time, Unemployed): ")
    marital_status = input("Enter marital status (Single, Married, Divorced): ")
    has_cosigner = input("Has a cosigner? (Yes/No): ")
    has_dependents = input("Has dependents? (Yes/No): ")
    loan_purpose = input("Enter loan purpose (Auto, Business, Other): ")

    # Return all input as a dictionary
    return {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'InterestRate': interest_rate,
        'Education': education,
        'LoanTerm': loan_term,
        'MonthsEmployed': months_employed,
        'EmploymentType': employment_type,
        'MaritalStatus': marital_status,
        'HasCoSigner': has_cosigner,
        'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose
    }

def preprocess_input(user_input):
    """Preprocess the user input to match the model's expectations."""
    
    # Convert binary features
    user_input['HasCoSigner'] = 1 if user_input['HasCoSigner'].lower() == 'yes' else 0
    user_input['HasDependents'] = 1 if user_input['HasDependents'].lower() == 'yes' else 0
    
    # Apply label encoding to categorical variables
    categorical_vars = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
    for var in categorical_vars:
        le = label_encoders[var]
        user_input[var] = le.transform([user_input[var]])[0]  # Encode and get the first element

    # Arrange input features in the correct order as per the training data
    feature_order = [
        'Age', 'Income', 'LoanAmount', 'InterestRate', 'Education', 'LoanTerm',
        'MonthsEmployed', 'EmploymentType', 'MaritalStatus', 'HasCoSigner',
        'HasDependents', 'LoanPurpose'
    ]
    
    # Extract values in the correct order
    input_values = [user_input[feature] for feature in feature_order]
    
    # Convert to numpy array and reshape for prediction
    return np.array(input_values).reshape(1, -1)

def predict_default_probability(model, processed_input):
    """Predicts the probability of loan default."""
    probability = model.predict_proba(processed_input)[:, 1]  # Get the probability of default (class 1)
    return probability[0]

if __name__ == "__main__":
    # Step 1: Get user input
    user_input = get_user_input()

    # Step 2: Preprocess the input
    processed_input = preprocess_input(user_input)

    # Step 3: Predict the probability of defaulting
    predicted_probability = predict_default_probability(model, processed_input)

    # Step 4: Output the result
    print(f"The predicted probability of defaulting on the loan is: {predicted_probability * 100:.2f}%")
