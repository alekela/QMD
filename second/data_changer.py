with open("Aluminum_data.txt") as f:
    title = f.readline()
    data = f.readlines()

new_data = []
for line in data:
    if line.strip().split()[0] == '1.000':
        new_data.append(line)

with open("Aluminum_data_ready.txt", 'w') as f:
    f.write(title)
    for line in new_data:
        f.write(line)