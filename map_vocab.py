f = open("vocab.txt", "r")
line = f.readline().split('\n')[0]
i = 0
d = {}
while(line != ""):
    d[line] = i
    i += 1
    line = f.readline().split('\n')[0]
f.close()

f = open("w2id.txt","w")
f.write( str(d) )
f.close()