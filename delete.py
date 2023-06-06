import os

file = "conflicts.txt"

to_remove = []

with open(file, "r") as f:
    lines = f.readlines()
    for line in lines:
        if not line.startswith("CONFLICT"):
            continue
        line = line.replace("CONFLICT (modify/delete):", "")
        idx = line.find(" deleted in nikhil/t2t-lite")
        file_name = line[:idx]
        to_remove.append(file_name)
        os.system(f"rm {file_name}")

print(to_remove)
