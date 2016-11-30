# -*- coding: utf-8 -*-
"""
This file appends <q> and </q> to the queries in the different session files. 
It results in sessions that contain more than one unique query.
"""

import utils
from sessionizer import Sessionizer
from datetime import datetime

# Train data
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading train sessions...]" % time)
snizer = Sessionizer('../data/tr_session')
sessions = snizer.get_sessions_with_numbers()
print("[Loaded %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Adding start and stop symbols...]" % time)
sessions = utils.append_start_stop_num(sessions, 'tr_session')
print("[The result is %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))

# Validation data
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading validation sessions...]" % time)
snizer = Sessionizer('../data/val_session')
sessions = snizer.get_sessions_with_numbers()
print("[Loaded %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Adding start and stop symbols...]" % time)
sessions = utils.append_start_stop_num(sessions, 'val_session')
print("[The result is %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))

# Test data
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading test sessions...]" % time)
snizer = Sessionizer('../data/test_session')
sessions = snizer.get_sessions_with_numbers()
print("[Loaded %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Adding start and stop symbols...]" % time)
sessions = utils.append_start_stop_num(sessions, 'test_session')
print("[The result is %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))