import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load the trained model
with open('knnbasic_model.pkl', 'rb') as f:
    algo_KNNBasic = pickle.load(f)

# Load the data
new_df = pd.read_csv('Amazonproducts.csv')

# Get the unique user IDs and product IDs
all_user_ids = list(new_df['userId'].unique())
train_user_ids = [algo_KNNBasic.trainset.to_raw_uid(i) for i in range(algo_KNNBasic.trainset.n_users)]

# Function to generate recommendations for a given user
def generate_recommendations(user_id):
    user_index = train_user_ids.index(user_id)
    uid = train_user_ids[user_index]
    items_purchased = algo_KNNBasic.trainset.ur[algo_KNNBasic.trainset.to_inner_uid(uid)]
    KNN_Product = algo_KNNBasic.get_neighbors(items_purchased[0][0], 15)
    recommendedation_lits = []
    for product_iid in KNN_Product:
        if not product_iid in items_purchased[0]: 
            purchased_item = algo_KNNBasic.trainset.to_raw_iid(product_iid)
            recommendedation_lits.append(purchased_item)
    return recommendedation_lits

# Streamlit app
st.title("Amazon Product Recommendation System")

st.write("Select a user ID to generate recommendations:")
user_id = st.selectbox("User ID", train_user_ids)

if st.button("Generate Recommendations"):
    recommendations = generate_recommendations(user_id)
    st.write("Recommended products for user", user_id, "are:")
    st.write(recommendations)
