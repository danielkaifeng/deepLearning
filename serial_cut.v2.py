
#from sys import argv

def normalization(seq):
	seq = [int(x) for x in seq]
	x_max = max(seq)
	x_min = min(seq)
	new_seq = 0.01 * [( x - x_min )/( x_max - x_min ) for x in seq] 
	
	return seq

ecgfile = "/data/run/daniel/project/02_samePerson/s400.index/get_tar/ecg_mat2.csv"
atrfile = "/data/run/daniel/project/02_samePerson/s400.index/get_tar/atr_mat2.csv"
annfile = "/data/run/daniel/project/02_samePerson/s400.index/get_tar/atrann_mat2.csv"


atr_out = ""
ecg_out = ""

cover_increment = 10 * 512
step_increment = 1 * 512

	

with open(atrfile,'r') as f1:
	atr_txt = f1.readlines()
with open(ecgfile,'r') as f1:
	ecg_txt = f1.readlines()
with open(annfile,'r') as f1:
	ann_txt = f1.readlines()

control = 0
person = 0 
for n in range(len(atr_txt)):
	ID = ecg_txt[n].split(',')[0]
	if n == 0: 
		old = ID
		print ID + ',%d' % person
	if ID != old:
		person += 1
		print ID + ',%d' % person
	old = ID
	
	ecg_data = ecg_txt[n].strip().split(',')[1:]
	atr_data = atr_txt[n].strip().split(',')[1:]
	ann_data = ann_txt[n].strip().split(',')[1:]
	
	for step in range(200):
		st = step*step_increment
		ed = step*step_increment + cover_increment

		x_seq = ecg_data[st:ed]
		y_seq = [ann_data[x] for x in range(len(ann_data)) if int(atr_data[x]) > st and int(atr_data[x]) < ed]

		if '30' in y_seq or len(y_seq)==0 or '14' in y_seq: # or len(y_seq) > 20 #or '14' not in y_seq 
			continue
		if len(x_seq) != cover_increment:
			continue

    filter = [int(atr[x]) for x in range(len(atr)) if anno[x]!=14]
    if len(filter) < 10: continue

    beat = 1
    for index in filter:
        if index < 85 or len(seq)-index < 160: continue 
        stR = index - 85
        edR = index + 170

        name = ID + '_' + str(beat)
        R_out += name + '\t' + '\t'.join(seq[stR:edR]) + '\n'


#		atr_out += ID + '_' + str(step) + ',' + ','.join(y_seq) + '\n'

		ecg_out += ','.join(x_seq) + ',%d\n' % person

		if (step+1)*step_increment + cover_increment > len(ecg_data): 
			break

#	if control > 30: break
#	control += 1

with open("ecg_10S.mat",'w') as f2:
	f2.writelines(ecg_out)
#with open("ecg_atr.mat",'w') as f2:
#	f2.writelines(atr_out)


"""
with open(argv[1],'r') as f1:
#	header = f1.readline().strip().split(',')
	txt = f1.readlines()

out = []
for line in txt:
	line = line.strip()
	l = line.split(',')
	ID = l[-1]
	data = l[0:-1]

#	for n in range(512,5120,512):
#		new_data = data[n:] + data[:n] + [ID]
#		out.append(','.join(new_data))
	for n in range(10):
		new_data = [str(float(x)*(1+random.randint(1,8)/100.)) for x in data] + [ID]
		out.append(','.join(new_data))

	out.append(line)

#header[0] = str(len(out))
#out = [','.join(header)] + out 
"""
