from features.adj import ADJ
import time

start = time.time()
print "adj:",
adj = ADJ()
print time.time() - start


start = time.time()
print "finding:",
adj.find_suitable_sessions()
print time.time() - start
