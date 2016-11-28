# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:50:50 2016

@author: Jaimy
"""

import utils
from sessionizer import Sessionizer
from datetime import datetime

# Train data
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading sessions...]" % time)
snizer = Sessionizer('../data/tr_session')
sessions = snizer.get_sessions_with_numbers()
print("[Loaded %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Adding start and stop symbols...]" % time)
sessions = utils.append_start_stop_num(sessions, 'tr_session')