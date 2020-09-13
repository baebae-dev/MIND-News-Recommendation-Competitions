import zipfile
import os

data_path = '../data/test'
f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
print('zip the file', os.path.join(data_path, 'prediction.txt'))
f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
f.close()

print('zippped file ', os.path.join(data_path, 'prediction.zip'))