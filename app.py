from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load the ML model
dictionary = pickle.load(open('./models/model.pkl', 'rb'))
model = dictionary['xgboost_model']
scaler = dictionary['standardscaler']

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        try:
            
            dependents = float(request.form.get('dependents'))
            graduated = request.form.get('graduated')
            graduated = 1 if graduated == 'GRADUATED' else 0
            selfi = request.form.get('self')
            selfi = 1 if selfi.lower() == 'yes' else 0
            income = int(request.form.get('income'))
            loan = int(request.form.get('loan'))
            loan_term = int(request.form.get('loan_term'))
            cibil = int(request.form.get('cibil'))
            resident = int(request.form.get('resident'))
            commercial_assets_value = int(request.form.get('commercial'))
            luxury_assets_value = int(request.form.get('luxury'))
            bank_assets_value = int(request.form.get('bank'))

            # Print the received data for debugging
            debt_to_income_ratio = income/loan
            income_ratio = (income*loan_term)/loan
            print(f"Received data: {dependents}, {graduated}, {selfi}, {income}, {loan}, {loan_term}, {cibil}, {resident}, {commercial_assets_value}, {luxury_assets_value}, {bank_assets_value},{debt_to_income_ratio},{income_ratio}")
     
            test = np.array([dependents, graduated, selfi, income, loan, loan_term, cibil, resident, commercial_assets_value, luxury_assets_value, bank_assets_value,debt_to_income_ratio,income_ratio])
            # Predict using the loaded model

            test= test.reshape(-1,1).T
            test_scaled =  scaler.transform(test)
            result = model.predict(test_scaled)
            result_text = 'Approved' if result[0] == 1 else 'Not Approved'

            # Print the result for debugging
            print(f"Prediction result: {result_text}")

            return render_template('index.html', results=result_text)
        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('index.html', results="Error processing the request")

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8080, debug=True)