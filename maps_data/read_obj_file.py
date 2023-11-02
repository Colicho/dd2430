import csv

file_to_read = "medium_receiver"
input_file = f'{file_to_read}.obj'
output_file = f'{file_to_read}.csv'

vertex_data = []
current_object = None

with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('o '):
            if current_object is not None:
                vertex_data.append(current_object)
            current_object = []
        elif line.startswith('v '):
            vertex = line.split()
            current_object.append((float(vertex[1]), float(vertex[2]), float(vertex[3])))

if current_object is not None:
    vertex_data.append(current_object)

with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X', 'Y', 'Z'])

    for obj in vertex_data:
        if len(obj) > 0:
            csv_writer.writerow(obj[0])

print(f"Data written to {output_file}")
