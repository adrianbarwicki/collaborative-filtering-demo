import numpy as numpy
import pandas as pd

# custom
import recommendation_helpers

## Reading data
cols_names = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./u.data', sep='\t', names=cols_names, encoding='latin-1')

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]


print str(n_users) + ' users | ' + str(n_items) + ' items'

# Construct user-item matrix
user_item_matrix = numpy.zeros((n_users, n_items))

# Fill out the user-item matrix
for row in ratings.itertuples():
    user_item_matrix[row[1]-1, row[2]-1] = row[3]

# Calculating sparcity: How many entries of the user_item_matrix are defined?
sparsity = float(len(user_item_matrix.nonzero()[0]))
sparsity /= (user_item_matrix.shape[0] * user_item_matrix.shape[1])
sparsity *= 100

print 'Sparsity: {:4.2f}%'.format(sparsity)

# We will split our data into training and test sets by removing 10 ratings per user from the training set and placing them in the test set.
train, test = recommendation_helpers.train_test_split(user_item_matrix)

# measure distance L2
user_similarity = recommendation_helpers.calc_similarity(train, kind='user')

# we predict with an average over all users' and display it's prediction error
simple_user_prediction = recommendation_helpers.predict_simple(train, user_similarity, 'user')
print 'Simple User-based CF MSE: ' + str(recommendation_helpers.get_mse(simple_user_prediction, test))

# we predict with an average over the k-most similar users' and display it's prediction error
topk_user_prediction = recommendation_helpers.predict_topk(train, user_similarity, kind='user', k=40)
print 'Top-k User-based CF MSE: ' + str(recommendation_helpers.get_mse(topk_user_prediction, test))



