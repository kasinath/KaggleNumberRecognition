import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

 


def loadTrainFile():
	csv_file_object = csv.reader(open('data/train.csv', 'rb')) 
	header = csv_file_object.next()	
	data=[]                          # Create a variable called 'data'.
	for row in csv_file_object:      # Run through each row in the csv file,
		data.append(row)             # adding each row to the data variable
	data = (np.array(data)).astype(np.int)
	return data

def loadTestFile():
	csv_file_object = csv.reader(open('data/test.csv', 'rb')) 
	header = csv_file_object.next()	
	data=[]                          # Create a variable called 'data'.
	for row in csv_file_object:      # Run through each row in the csv file,
		data.append(row)             # adding each row to the data variable
	data = (np.array(data)).astype(np.int)
	return data
	

def main():
	print("Loading Training data")
	data = loadTrainFile();
	print("Loading Testing data")
	test_data = loadTestFile();
	print("Training")
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(data[0::,1::],data[0::,0])
	print("Predicting")
	op = forest.predict(test_data)
	print("Generating OP file")
	fOut = open('data/op.csv','w')
	fOut.write("ImageId,Label\n")
	count = 1
	for line in op:
		fOut.write(str(count) + "," + str(line) + "\n")
		count=count+1
	fOut.close()

	
	
	
if __name__=="__main__":
	main()
