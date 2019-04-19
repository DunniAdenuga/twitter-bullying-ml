

f = open('data.csv', 'r')
g = open("tweet_id", "w")

for line in f:
    y = line.split(',')
    g.write(y[0] + "," + y[2] + "\n")

g.close()
f.close()
