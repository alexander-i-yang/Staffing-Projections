
in_file = open("output.csv", "r")
out_file = open("output2.csv", "w")

out_file.write('\n'.join(list(filter(("").__ne__, in_file.read().split("\n")))))

in_file.close()
out_file.close()