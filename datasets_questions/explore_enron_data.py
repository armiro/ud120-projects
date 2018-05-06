#!/C:/Python27
# compatible with Python2+
""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# getting into enron data:
print 'Number of features per each person:', len(enron_data['METTS MARK'])
print 'Number of person records in the dataset:', len(enron_data)
print 'type of our dataset is:', type(enron_data)


# number of POIs in the data:
numPOI = 0
for person_name in enron_data:
    if enron_data[person_name]['poi'] is True:
        numPOI += 1
print 'Number of persons of interest:', numPOI


# total number of POIs:
fileName = "../final_project/poi_names.txt"
fileIn = open(fileName, 'r')
fileIn.readline()
fileIn.readline()

numPOIsInTotal = 0
for line in fileIn:
    numPOIsInTotal += 1
print 'Total number of POIs:', numPOIsInTotal

# total value of stock belonging to James Prentice
for i in range(1, len(enron_data)):
    if enron_data.keys()[i] == 'PRENTICE JAMES':
        print 'Total val. stock of J.Prentice', enron_data[enron_data.keys()[i]]['total_stock_value']

# number of email from Wesley Colwell to POIs
for i in range(1, len(enron_data)):
    if enron_data.keys()[i] == 'COLWELL WESLEY':
        print 'Num emails from W.Colwell to POIs:', enron_data[enron_data.keys()[i]]['from_this_person_to_poi']

# value of stock options exercised by Jeffrey K Skilling
for names in enron_data.keys():
    if names == 'SKILLING JEFFREY K':
        print 'Value of exercised stock by J.Skilling:', enron_data[names]['exercised_stock_options']

# who took home the most money?!
highestMoney = 0
theName = ''
for names in enron_data.keys():
    if names == 'FASTOW ANDREW S' or names == 'SKILLING JEFFREY K' or names == 'LAY KENNETH L':
        if enron_data[names]['total_payments'] > highestMoney:
            highestMoney = enron_data[names]['total_payments']
            theName = names
        else:
            pass
print 'His/her name is:', theName, 'And the total payment was:', highestMoney

# dealing with unfilled data (multiple quizzes)
quantifiedSalary = 0
quantifiedEmail = 0
quantifiedPayments = 0
poiWithoutPayments = 0

for names in enron_data.keys():
    if enron_data[names]['salary'] != 'NaN':
        quantifiedSalary += 1
    if enron_data[names]['email_address'] != 'NaN':
        quantifiedEmail += 1
    if enron_data[names]['total_payments'] != 'NaN':
        quantifiedPayments += 1
    if enron_data[names]['poi'] is True:
        if enron_data[names]['total_payments'] == 'NaN':
            poiWithoutPayments += 1

print 'Num of quantified salary persons:', quantifiedSalary
print 'Num of quantified email persons:', quantifiedEmail
print 'Num of quantified payments persons:', quantifiedPayments
print 'Num of unquantified payments POIs:', poiWithoutPayments

# finished the tasks in this chapter/
