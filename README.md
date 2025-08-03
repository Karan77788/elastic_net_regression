# elastic_net_regression
# Customer Lifetime Value Estimator
This project is a machine learning web application built with Elastic Net Regression and Flask. It predicts the Customer Lifetime Value (CLV) based on demographics and behavioral inputs such as income, membership duration, average transaction value, and more.

 # Machine Learning Model
Algorithm Used: ElasticNet Regression
Why Elastic Net?
Elastic Net combines Lasso and Ridge regression. It's especially useful when:

We have many features (some of which may be correlated)

We want to reduce overfitting

We want both regularization and feature selection

 # Features Used
Feature	Description
Age	Age of the customer
Gender	Male / Female
Income	Annual income (in ₹)
MembershipDuration	Years since registered
TotalTransactions	Total number of transactions
AvgTransactionValue	Average value of each transaction
ProductCategory	Electronics, Fashion, Home, etc.
EngagementScore	Score (1–10) based on activity and behavior

Target: Customer Lifetime Value (CLV) in ₹

 # Project Structure
 ```
Customer_Lifetime_Value_Estimator/
│
├── model/
│   └── elasticnet_clv_model.pkl         # Trained ML model
│
├── static/
│   └── style.css                        # Custom CSS styling
│
├── templates/
│   ├── index.html                       # Input form
│   └── result.html                      # Display prediction
│
├── clv_data.csv                         # Dataset (30 rows or more)
├── train_model.py                       # ML training script
├── app.py                               # Flask backend
└── README.md                            # Project README
```
# Clone the repository
```
git clone https://github.com/yourusername/clv-estimator.git
```
cd clv-estimator
# Install dependencies
```
pip install -r requirements.txt
```
# Train the model
```
python train_model.py
```
# Start the Flask app
```
python app.py
```
Open in browser
```
Navigate to http://127.0.0.1:5000
```
 # Example Input
Age: 30

Gender: Female

Income: ₹800000

Membership Duration: 3.5 years

Total Transactions: 45

Avg Transaction Value: ₹1700

Product Category: Fashion

Engagement Score: 8

 Output: Predicted CLV: ₹205,000

# ScreenShot 
![alt text](<Screenshot 2025-08-03 100056.png>)
![alt text](<Screenshot 2025-08-03 100142.png>)
![alt text](<Screenshot 2025-08-03 100153.png>)

