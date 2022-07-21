def paramnames_file(path, params):
    lines = [f"{p}   {q}\n" for (p, q) in params]
    with open(f"{path}.paramnames", "w") as f:
        f.writelines(lines)
