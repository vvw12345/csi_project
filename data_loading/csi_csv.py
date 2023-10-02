import csv
def save_csv(csi_list,filepath):
    #csi_list = list(map(get_scale_csi,bf))
    with open(filepath,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in csi_list:
            writer.writerow(row)
    