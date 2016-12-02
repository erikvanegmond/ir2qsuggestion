from features.adj import ADJ
from datetime import datetime

start = datetime.now()
time = start.strftime('%d-%m %H:%M:%S')
print('[%s: Creating ADJ...]' % time)
adj = ADJ()
print('[It took %s seconds.]' % (start - datetime.now()).seconds)

start = datetime.now()
time = start.strftime('%d-%m %H:%M:%S')
print('[%s: Searching for suitable train queries...]' % time)
adj.find_suitable_sessions()
print('[It took %s seconds.]' % (start - datetime.now()).seconds)
