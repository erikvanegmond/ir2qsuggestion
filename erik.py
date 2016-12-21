adj = 0.7690

for e in [[adj, 0.7825, 0.7854], [adj, 0.7852, 0.7833], [adj, 0.7939, 0.7928]]:
    for i, score in enumerate(e[1:]):
        s = []
        for c in e[:i+1]:
            percent = ((score/c)-1)*100
            sp = "$%.1f%s$" % (percent,  "\\%")
            s.append(sp)
        print score, "/".join(s)

