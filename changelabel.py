import os

dir = './1_dataprep/data_images/test'

for file in os.listdir(dir):
	if file.endswith('.txt'):
		file_pth = os.path.join(dir, file)
		with open(file_pth, 'r') as f:
			lines = f.readlines()
		corrected  = []
		for line in lines:
			segments = line.split()
			# Change the first character down
			# Fixes nc labeling issue
			if segments:
				segments[0] = '0'
				corrected.append(' '.join(segments) + '\n')
		with open(file_pth, 'w') as f:
			f.writelines(corrected) 
print("0")