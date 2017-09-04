
from sys import argv
import random

with open(argv[1],'r') as f1:
#	header = f1.readline().strip().split(',')
	txt = f1.readlines()

out = []
for line in txt:
	line = line.strip()
	l = line.split(',')
	ID = l[-1]
	data = l[0:-1]

	for n in range(512,5120,512):
		new_data = data[n:] + data[:n] + [ID]
		out.append(','.join(new_data))
#	for n in range(10):
#		new_data = [str(float(x)*(1+random.randint(1,8)/100.)) for x in data] + [ID]
#		out.append(','.join(new_data))

	out.append(line)

#header[0] = str(len(out))
#out = [','.join(header)] + out 

print '\n'.join(out)

