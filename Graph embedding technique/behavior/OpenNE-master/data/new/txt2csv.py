#  @author rzh
#  @create 2021-04-25 20:41

input_file = open('DNA RBP HOST pairs.csv', 'r')
output_file = open('DNARBPHOSTALL.txt', 'w')
# input_file.readline() # skip first line
for line in input_file:
    output_file.write(' '.join(line.strip().split(',')) + '\n')
input_file.close()
output_file.close()