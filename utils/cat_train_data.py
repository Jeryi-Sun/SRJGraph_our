import os
root_data_path = '../../reco_search_data/'
data_path = ["train_data_reco_1.txt", "train_data_reco_2.txt", "train_data_reco_3.txt", 'train_data_reco_4.txt']
all_train_data = []
for file_name in data_path:
    with open(os.path.join(root_data_path, file_name), 'r') as f:
        for line in f:
            all_train_data.append(line)

with open(os.path.join(root_data_path, "train_final_reco_data.txt"), 'w') as f:
    for line in all_train_data:
        f.writelines(line)           
 

