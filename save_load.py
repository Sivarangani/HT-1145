import pickle

def save(name, val):
    # Open a file and use dump()
    with open('./pkl/'+name+'.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(val, file)

def load(name):
    # Open the file in binary mode
    with open('./pkl/'+name+'.pkl','rb') as file:
        # Call load method to deserialze
        myvar = pickle.load(file)
    return myvar