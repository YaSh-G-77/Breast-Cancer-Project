# Breast Cancer Classification

## About the Project

This project uses Machine Learning to classify breast tumor cases as **Benign** or **Malignant** based on input features.
The project includes the complete machine learning workflow, from data analysis and preprocessing to model training and prediction.

## Project Structure
```text
breast-cancer-project/
├── breastcancer.ipynb   # Jupyter Notebook containing the ML workflow
├── app.py               # Application for making predictions
├── model.pkl            # Trained machine learning model
├── scaler.pkl           # Saved feature scaler
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation
```
## Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Machine Learning

## Machine Learning Workflow
The project follows these general steps:
1. Load the dataset
2. Explore and understand the data
3. Perform data preprocessing
4. Split the data into training and testing sets
5. Scale the features
6. Train the machine learning model
7. Evaluate the model
8. Save the trained model and scaler
9. Use the saved model for new predictions

## Model Output
The application predicts one of two classes:
* **Benign**
* **Malignant**

These predictions are intended for **educational and demonstration purposes only** and should not be used as a medical diagnosis.

## How to Run the Project
### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/breast-cancer-project.git
```
### 2. Open the project directory
```bash
cd breast-cancer-project
```
### 3. Install the required libraries
```bash
pip install -r requirements.txt
```
### 4. Run the application
Use the command appropriate for the application in `app.py`.
For example, if it is a standard Python application:
```bash
python app.py
```
If it is a Streamlit application:
```bash
streamlit run app.py
```
## Files
### `breastcancer.ipynb`
Contains the exploratory analysis, preprocessing, model training, evaluation, and other machine learning work.
### `app.py`
Contains the application used to make predictions using the trained model.
### `model.pkl`
Contains the trained machine learning model saved using Python's pickle format.
### `scaler.pkl`
Contains the feature-scaling object used to preprocess input data before prediction.
## Disclaimer

This project is created for learning and demonstration purposes. It is not a medical diagnostic system and should not be used to make real-world medical decisions.
