## Collaborative Filtering
Collaborative filtering is a method used by recommender systems based on the idea that your preferences / ratings / behaviour can be predicted from actions of people similar to you.

This demo intends to demonstrate the basic principle of Collaborative-Based Recommender Systems.

## Sample data
The movieLens data is in the format:
```
       user_id  movie_id  rating  unix_timestamp
0          196       242       3       881250949
1          186       302       3       891717742
2           22       377       1       878887116
...
```
## Steps
1. Transform the data into user-item utility matrix
2. Compute sparcity of the user-item matrix
3. Compute similarity matrix for every user pair
4. Cross-validation: Split the data into training and test data
5. Compute similarity for every user pair (we use cosine similarity algorithm)
6. Compute missing entries in the user-item utility matrix using all user's ratings
7. Compute missing entries in the user-item utility matrix using all the most similar users' ratings

## Dependencies
pandas, numpy, sklearn

## Read more
[RECOMMENDATION SYSTEMS: SIMPLE THEORY OF COLLABORATIVE FILTERING](http://adrianbarwicki.com/2017/06/16/collaborative-filtering-recommendation-systems-simple-theory/)