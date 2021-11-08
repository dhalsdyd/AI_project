import csv
import numpy as np
'''일반적으로
f = open('train.csv','r',encoding='utf-8')
rdr = csv.reader(f)
print(rdr[0])
f.close()
'''
csv_data =  np.genfromtxt(fname = 'train.csv', delimiter=',',dtype = np.float32)


x_data = [v[0] for v in csv_data]
y_data = [v[1] for v in csv_data]

