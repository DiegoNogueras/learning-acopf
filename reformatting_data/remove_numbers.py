file_with_nums = "/Users/nogueras1/Documents/ACOPF_Workspace/reformatting_data/case300/case300_data.txt"
new_file = "case300.txt"

with open(file_with_nums, 'r') as file:
    data = file.readlines()

for i in range(len(data)):
    data[i] = data[i][4:].lstrip()

with open(new_file, 'a') as file:
    file.writelines(data)
