# Airline Fare Prediction Using Machine Learning

This project predicts airline ticket prices using machine learning, looking at factors like flight schedules, destinations, and seasonality. It uses models like KNN, Linear Regression, and Random Forest, with a Django-based platform for users to get fare predictions. Future work will improve accuracy and user experience.

## Project Description

This project predicts airline ticket prices using machine learning by analyzing factors such as flight schedules, destinations, seasonality, and the number of stops. The system uses three datasets to train and test models that estimate ticket prices based on these features. The project uses models like KNN Regression, Linear Regression, Decision Tree Regression, Stacking Tree Regression, and Random Forest Regression. Random Forest performed the best in terms of accuracy.

A web-based platform built with Django allows users to enter flight details—like departure and destination cities, travel time, and number of stops—to get fare predictions. The platform also includes an Admin Dashboard, where admins can track predictions and compare model performances through charts.

The project demonstrates how machine learning can help predict flight prices, making it easier for travelers to make informed decisions. While the system is useful, it can be improved by adding more factors like seat availability, public holidays, and baggage allowance. Future work will focus on refining the features, expanding the dataset, improving the model, and making the user interface even better.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/airline-fare-prediction.git
   cd airline-fare-prediction
   ```

2. **Set up a Python virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   Make sure you have `requirements.txt` in the project folder and install the required libraries.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Django server**:

   ```bash
   python manage.py runserver
   ```

   You can access the system in your browser at `http://127.0.0.1:8000/`.

### Requirements

[requirements file](requirements.txt)

- `asgiref==3.6.0`
- `Django==4.1.5`
- `joblib==1.2.0`
- `mysql==0.0.3`
- `mysqlclient==2.1.1`
- `numpy==1.24.1`
- `pandas==1.5.3`
- `Pillow==9.4.0`
- `python-dateutil==2.8.2`
- `pytz==2022.7.1`
- `scikit-learn==1.2.0`
- `scipy==1.10.0`
- `six==1.16.0`
- `sklearn==0.0.post1`
- `sqlparse==0.4.3`
- `threadpoolctl==3.1.0`
- `tzdata==2022.7`

### Technology Stack and Models Used

- **Programming Language**: Python 3.10  
- **Web Framework**: Django  
- **Machine Learning**: Scikit-learn  
- **Frontend Technologies**: HTML5, CSS3, JavaScript  
- **Database**: MySQL  
- **Development Tools**: Visual Studio Code  
- **Version Control**: Git & GitHub  

### Models Used for Prediction:
The following machine learning models were utilized to predict airline ticket prices:  
1. **KNN Regression**  
2. **Linear Regression**  
3. **Decision Tree Regression**  
4. **Stacking Tree Regression**  
5. **Random Forest Regression**  