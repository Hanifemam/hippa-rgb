evn_var_dict = {}
with open("/home/hemamgholizadeh/hippa-rgb/env_var.txt") as f:
    env_var = f.read()
    for line in env_var.split("\n"):
        elements = line.split("=")
        if len(elements) == 2:
            key = elements[0].strip()
            value = elements[1].strip().replace('"', "").replace("'", "")
            evn_var_dict[key] = value
print(evn_var_dict)