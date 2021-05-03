import numpy as np
import os
from pathlib import Path
from zipfile import ZipFile
import zipfile
import zlib

'''
Run this script in the folder that you want to zip the files too
'''

def chunks(l, n):
    """Yield successive n-sized chunks from lst.
    from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

file_array = []
for file in os.listdir(Path().absolute()):
    # only look at .pickle files
    if file.endswith(".pickle"):
        file_array.append(file)
        
file_array = list(chunks(file_array, 50))
np.shape(file_array)

index_counter = 0
file_name = '{}.zip'.format(str(index_counter))
for i in file_array:
    print(index_counter)
        
    with ZipFile(file_name, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for f in i:
            zip_file.write(f)
    index_counter += 1
    file_name = '{}.zip'.format(str(index_counter))