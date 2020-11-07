import json, sys
import numpy as np

if len(sys.argv)<2:
    print("Please, specify a json file")
    exit(0)

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

distance_to_goal = []
norm_navigation_time = []
norm_distance_traveled = []
navigation_time = []
distance_traveled = []
chc = []
intrusions_intimate = []
intrusions_personal = []
intrusions_interactions = []
min_distance_to_humans = []
for d in data:
    norm_navigation_time.append(d['navigation_time']/d['distance_to_goal'])
    norm_distance_traveled.append(d['distance_traveled']/d['distance_to_goal'])
    distance_to_goal.append(d['distance_to_goal'])
    navigation_time.append(d['navigation_time'])
    distance_traveled.append(d['distance_traveled'])
    chc.append(d['chc2'])
    intrusions_intimate.append(d['number_of_intrusions_intimate']*100./d['number_of_steps'])
    intrusions_personal.append(d['number_of_intrusions_personal']*100./d['number_of_steps'])
    intrusions_interactions.append(d['number_of_intrusions_in_interactions']*100./d['number_of_steps'])
    min_distance_to_humans.append(d['minimum_distance_to_human'])

print("normalized navigation time", np.mean(norm_navigation_time), np.std(norm_navigation_time))
print("normalized distance traveled", np.mean(norm_distance_traveled), np.std(norm_distance_traveled))
print("navigation time", np.mean(navigation_time), np.std(navigation_time))
print("distance traveled", np.mean(distance_traveled), np.std(distance_traveled))
print("chc", np.mean(chc), np.std(chc))
print("percentage intrusions intimate", np.mean(intrusions_intimate), np.std(intrusions_intimate))
print("percentage intrusions personal", np.mean(intrusions_personal), np.std(intrusions_personal))
print("percentage intrusions interactions", np.mean(intrusions_interactions), np.std(intrusions_interactions))
print("minimum distance to human", np.mean(min_distance_to_humans), np.std(min_distance_to_humans))
