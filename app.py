from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("elasticnet_clv_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = {
            "Age": float(request.form["age"]),
            "Gender": request.form["gender"],
            "Income": float(request.form["income"]),
            "MembershipDuration": float(request.form["membership"]),
            "TotalTransactions": int(request.form["transactions"]),
            "AvgTransactionValue": float(request.form["avg_value"]),
            "ProductCategory": request.form["category"],
            "EngagementScore": float(request.form["engagement"])
        }
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)
    return render_template("index.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)
