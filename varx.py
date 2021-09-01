import os
path = 'Violation_A6'
files = os.listdir(path)
i=1
for file in files:
   os.rename(os.path.join(path, file), os.path.join(path, 'V_A6_' + str(i) + '.pdf'))
   i+=1