import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class RecommendationModel:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_df = None

    def load_data(self, file_path):
        # Load data from CSV file
        my_data_set = pd.read_csv(file_path)

        # Identify columns
        user_column = 'SUBSCRIPTION_ID'
        item_column = 'BUNDLE_NAME'
        activation_column = 'REVENUE'

        # Create a binary interaction column
        my_data_set['Interaction'] = my_data_set[activation_column].apply(lambda x: 1 if x > 0 else 0)

        # Create the User-Item Interaction Matrix
        self.user_item_matrix = my_data_set.pivot_table(values='Interaction', index=user_column, columns=item_column,
                                                        fill_value=0)

        # Convert the pivot table to a NumPy array
        user_item_matrix_np = self.user_item_matrix.to_numpy()

        # Calculate cosine similarity between all pairs of users
        user_similarity_matrix = cosine_similarity(user_item_matrix_np)

        # Convert the similarity matrix to a DataFrame
        self.user_similarity_df = pd.DataFrame(user_similarity_matrix, index=self.user_item_matrix.index,
                                               columns=self.user_item_matrix.index)

    def save_model(self, model_path):
        # Save the model to a file
        with open(model_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(model_path):
        # Load the model from a file
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    def get_users_to_activate_bundles(self, bundle_name, N=3, threshold=0.7):
        user_recommendations = {}
        for target_user in self.user_item_matrix.index:
            target_user_data = self.user_item_matrix.loc[target_user]
            if target_user_data[bundle_name] == 0:
                similar_users = self.user_similarity_df[target_user].sort_values(ascending=False).drop(target_user)
                similar_user_activation = self.user_item_matrix.loc[similar_users.index, bundle_name] == 1
                true_subscriptions = similar_user_activation[similar_user_activation].index
                similar_users_with_true_activation = similar_users[true_subscriptions]
                top_similar_users = similar_users_with_true_activation[
                    (similar_users_with_true_activation >= threshold)].head(N)
                user_recommendations[target_user] = top_similar_users.index.tolist()

        recommendations = list(user_recommendations.keys())
        recommendation_final = pd.DataFrame({'subscription_id': recommendations, 'bundle_to_recommend': bundle_name})
        return recommendation_final


# Example usage:
# model = RecommendationModel()
# model.load_data("./data_bundle.csv")
# model.save_model("bundle_recommendation_model.pkl")
