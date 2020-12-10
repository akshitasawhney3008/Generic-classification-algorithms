import pickle
import csv

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

with open('ListOfBestParamsGS(0.15-0.19).pkl', 'rb') as f:
    random_params_dict = pickle.load(f)

with open('ListOfBestParamsGS(0.20-0.24).pkl', 'rb') as f:
    random_params_dict1 = pickle.load(f)

with open('ListOfBestParamsGS(0.25-0.29).pkl', 'rb') as f:
    random_params_dict2 = pickle.load(f)

with open('ListOfBestParamsGS(0.30-0.40).pkl', 'rb') as f:
    random_params_dict3 = pickle.load(f)

with open('ListOfBestParamsGS(0.41-0.57).pkl', 'rb') as f:
    random_params_dict4 = pickle.load(f)


z = merge_dicts(random_params_dict,random_params_dict1, random_params_dict2, random_params_dict3, random_params_dict4)
print("hello")
count = 0
with open('ListOfBestParamsGS.csv', 'w') as csv_file:
    count+=1
    print(count)
    writer = csv.writer(csv_file)
    for key, value in z.items():
       writer.writerow([key, value])

with open('ListOfBestParamsGS.pkl', 'wb') as f:
    pickle.dump(z,f)

