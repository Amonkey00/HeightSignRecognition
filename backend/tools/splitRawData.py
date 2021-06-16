import os
from shutil import copy2


# Set file& save Path
file_dir = 'E:/PycharmProjects/test1/HeightLimitSign2021/outputs'
save_dir = 'F:/HeightLimit'

# Set the number of parts
part_num=3

# Get files from file_dir
files = os.listdir(file_dir)
# Compute the gapNum
gapNum=len(files)//part_num

index=0
for i, filename in enumerate(files):
    if i % gapNum==0:
        index += 1
        index = min(3,index)
    file_from=os.path.join(file_dir,filename)
    file_to=os.path.join(save_dir,str(index)+'/Signs')
    print(i)
    copy2(file_from,file_to)
