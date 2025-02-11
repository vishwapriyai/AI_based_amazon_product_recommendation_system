# Amazon Product Recommendation System

## Overview
This project implements a recommendation system for Amazon products using collaborative filtering techniques. The system utilizes user ratings to suggest products that a user may be interested in based on their past interactions and the preferences of similar users.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Surprise (for collaborative filtering)
- Streamlit (for web application)

## Dataset
The dataset used in this project is a subset of Amazon product ratings, which includes the following columns:
- `userId`: Unique identifier for each user
- `productId`: Unique identifier for each product
- `Rating`: Rating given by the user to the product
- `timestamp`: Time when the rating was given (not used in the final model)

## Installation
To run this project, you need to have Python installed on your machine. You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-surprise streamlit
```
## Usage

### Data Preparation
1. Place the dataset `Amazonproducts.csv` in the project directory.
2. The data is preprocessed in the `model.ipynb` notebook, which includes:
   - Loading the data
   - Cleaning the data
   - Performing exploratory data analysis (EDA)

### Model Training
1. The recommendation model is trained using the KNNBasic algorithm from the Surprise library.
2. The trained model is saved as `knnbasic_model.pkl`.

### Running the Streamlit App
1. Navigate to the directory containing `app.py`.
2. Run the following command in your terminal:
   ```bash
   streamlit run app.py

### Running the Streamlit App
This will start a local web server and open the application in your default web browser.

### Generating Recommendations
1. Select a user ID from the dropdown menu in the Streamlit app.
2. Click the "Generate Recommendations" button to see the recommended products for the selected user.

### Example
After selecting a user ID and clicking the button, the app will display a list of recommended products based on the user's past ratings and the ratings of similar users.

## Conclusion
This project demonstrates the implementation of a collaborative filtering recommendation system using user ratings. The model can be further improved by experimenting with different algorithms, tuning hyperparameters, or incorporating additional features.
