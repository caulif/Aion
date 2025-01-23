import pickle

initialization_values = (1, 8, 173569775688864, 5000999999999999)

with open("initialization_values", 'wb') as file:
    pickle.dump(initialization_values, file)
