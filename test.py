import pandas
from sklearn.ensemble import GradientBoostingClassifier
import pickle


pkl_filename = "cactus_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
score = pickle_model.predict([[1,2,1,1,1,1,2,1,1,1,2,1,2]])
print(score)