#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#


from genericworker import *

import os
import json
import subprocess
import time
import signal
import inspect
import numpy as np
import threading
import copy
import sys, os

from shapely.geometry import Polygon, Point

import datetime
import math
from PySide2 import QtGui, QtCore, QtWidgets

import PySide2.QtGui as QtGui
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import QSettings

import _pickle as pickle
import json
from PySide2.QtWidgets import (QLabel, QLineEdit, QPushButton, QApplication,
    QVBoxLayout, QDialog, QWidget)
from ui_configuration import *

sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../nets'))

from gaussianSpaces import GaussianSpaces, Person
import graph2imageUp
from graph2imageUp import SNGNN2D, SNScenario
import cv2

def rad_to_deg(a):
    angle = a*180./math.pi
    while angle >= 180.:
        angle -= 360
    while angle <= -180.:
        angle += 360
    return angle

class PersonFields():
    def __init__(self, x, z, angle):
        self.x = x
        self.z = z
        self.angle = angle

class ObjectFields():
    def __init__(self, x, z, angle, space):
        self.x = x
        self.z = z
        self.angle = angle
        self.space = space


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, fullscreen, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
#        os.system('bash ../joystick.sh &')
        os.system('bash ../simulator.sh &')
        self.timer.timeout.connect(self.compute)

        self.gaussian_spaces = GaussianSpaces()

        self.img_mutex = threading.RLock()
        self.img = None
        self.proximity_threshold = 0.4
        self.robot_coordinates = [0,0]
        ###
        self.window = QWidget()
        self.button = QPushButton("enter")
        self.layout = QVBoxLayout()
        ###
        self.goal_coordinates = [100,100]
        self.Period = 100
        self.timer.start(self.Period)
        self.data = []
        self.goal_data = []
        self.wall_data = []
        self.interaction_data = []
        self.people_data = []
        self.object_data = []
        self.updates_list = []
        self.data_json = []
        self.speed_command = [0., 0., 0.]
        self.relations_dic = {}

        #Metrics
        self.init_metrics = True
        self.initial_time = 0.
        self.navigation_time = 0.
        self.last_robot_position = []
        self.distance_traveled = 0.
        self.interaction_areas = []

        self.configGui = Ui_confDialog()
        self.configWindow = QDialog()
        self.configGui.setupUi(self.configWindow)
        if fullscreen != "True":
            self.WINDOW_WIDTH = 1042
            self.WINDOW_HEIGHT = 819
            self.setFixedSize(self.WINDOW_WIDTH,self.WINDOW_HEIGHT)
        self.save_interval = 0.1 #
        self.last_saved_time = time.time()

        self.new_goal = False
        self.on_a_mission = True
        self.started_mission = False
        self.ui.configuration.clicked.connect(self.configuration_slot)
        self.ui.regenerate.clicked.connect(self.regenerate_slot)
        self.ui.quit.clicked.connect(self.quit_slot)
        self.ui.save_metrics.clicked.connect(self.save_metrics)

        self.configGui.nhumans_min.valueChanged.connect(self.min_nhumans_changed)
        self.configGui.nhumans_max.valueChanged.connect(self.max_nhumans_changed)
        self.configGui.nwandhumans_min.valueChanged.connect(self.min_nwandhumans_changed)
        self.configGui.nwandhumans_max.valueChanged.connect(self.max_nwandhumans_changed)
        self.configGui.ntables_min.valueChanged.connect(self.min_ntables_changed)
        self.configGui.ntables_max.valueChanged.connect(self.max_ntables_changed)
        self.configGui.nplants_min.valueChanged.connect(self.min_nplants_changed)
        self.configGui.nplants_max.valueChanged.connect(self.max_nplants_changed)
        self.configGui.nrelations_min.valueChanged.connect(self.min_nrelations_changed)
        self.configGui.nrelations_max.valueChanged.connect(self.max_nrelations_changed)
        
        if fullscreen == 'True':
            self.showFullScreen()

    @QtCore.Slot()
    def configuration_slot(self, active):
        if active:
            self.configWindow.show()
        else:
            self.configWindow.hide()

    @QtCore.Slot()
    def regenerate_slot(self):

        print('regenerate')
        self.omnirobot_proxy.setSpeedBase(0, 0, 0)
        self.i_frame = 0
        self.updates_list = []
        self.initial_positions = []
        self.initial_positions_objects = []
        self.initial_flag=0
        self.info()
        
        self.simulator_proxy.regenerate(scene=self.relations_dic)
        self.new_goal = False
        self.on_a_mission = True
        self.started_mission = False

        self.init_metrics = True


    @QtCore.Slot()
    def quit_slot(self):
        os.system('pwd')
        os.system('bash ../kill.sh &')


    @QtCore.Slot()
    def keyPressEvent(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Q:
                self.quit_slot()
            elif event.key() == QtCore.Qt.Key_R:
                self.regenerate_slot()


    @QtCore.Slot()
    def min_nhumans_changed(self, val):
        if val>self.configGui.nhumans_max.value():
            self.configGui.nhumans_max.setValue(val)

    @QtCore.Slot()
    def max_nhumans_changed(self, val):
        if val<self.configGui.nhumans_min.value():
            self.configGui.nhumans_min.setValue(val)

    @QtCore.Slot()
    def min_nwandhumans_changed(self, val):
        if val>self.configGui.nwandhumans_max.value():
            self.configGui.nwandhumans_max.setValue(val)

    @QtCore.Slot()
    def max_nwandhumans_changed(self, val):
        if val<self.configGui.nwandhumans_min.value():
            self.configGui.nwandhumans_min.setValue(val)

    @QtCore.Slot()
    def min_ntables_changed(self, val):
        if val>self.configGui.ntables_max.value():
            self.configGui.ntables_max.setValue(val)

    @QtCore.Slot()
    def max_ntables_changed(self, val):
        if val<self.configGui.ntables_min.value():
            self.configGui.ntables_min.setValue(val)

    @QtCore.Slot()
    def min_nplants_changed(self, val):
        if val>self.configGui.nplants_max.value():
            self.configGui.nplants_max.setValue(val)

    @QtCore.Slot()
    def max_nplants_changed(self, val):
        if val<self.configGui.nplants_min.value():
            self.configGui.nplants_min.setValue(val)

    @QtCore.Slot()
    def min_nrelations_changed(self, val):
        if val>self.configGui.nrelations_max.value():
            self.configGui.nrelations_max.setValue(val)

    @QtCore.Slot()
    def max_nrelations_changed(self, val):
        if val<self.configGui.nrelations_min.value():
            self.configGui.nrelations_min.setValue(val)


    def __del__(self):
        os.system('kill -15 `ps ax |grep simulator | grep config | awk \'{print $1}\'`')
#        os.system('kill -15 `ps ax |grep joystick | grep config | awk \'{print $1}\'`')
        print('SpecificWorker destructor')
        self.quit_slot()

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        self.show_image()

        if not self.on_a_mission or not self.new_goal:
            return

        proximity = math.sqrt(self.goal_coordinates[0] ** 2 + self.goal_coordinates[1] ** 2)
        
        if proximity <= self.proximity_threshold:
            self.on_a_mission = False

        self.update_metrics()

        scenario = self.get_scene()
        d = SNGNN2DData()
        d.gridSize = scenario.shape[0]
        d.areaSize = 800.
        d.g = scenario.reshape([1,d.gridSize*d.gridSize])[0]
        d.g = d.g.tolist()
        self.sngnn2d_proxy.gotgrid(d)
       

    def update_metrics(self):

        robot_position = [ [r.x, r.y] for r in self.object_data if r.id==-2]
        robot_angle = [ r.angle for r in self.object_data if r.id==-2]
        
        if self.init_metrics:
            self.initial_time = time.time()
            self.last_time = self.initial_time
            self.distance_to_goal = math.sqrt(self.goal_coordinates[0] ** 2 + self.goal_coordinates[1] ** 2)
            self.last_robot_position = robot_position[0]
            self.last_robot_angle = robot_angle[0]
            self.distance_traveled = 0.
            self.cum_heading_changes1 = 0.
            self.cum_heading_changes2 = 0.
            self.minimum_distance_to_human = 100.
            self.number_of_intrusions_intimate = 0
            self.number_of_intrusions_personal = 0
            self.number_of_intrusions_in_interactions = 0
            self.number_of_steps = 0
            self.area_computed = False
            self.room_area = 0
            self.init_metrics = False

        if not self.area_computed:
            if self.wall_data:
                room = []
                for wall in self.wall_data:
                    room.append([wall.x1, wall.y1])
                room.append([self.wall_data[-1].x2, self.wall_data[-1].y2])
                self.room_area = Polygon(room).area
                self.area_computed = True
           

        self.number_of_steps += 1
        current_time = time.time()
        self.navigation_time = current_time-self.initial_time

        self.distance_traveled += math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.last_robot_position, robot_position[0])]))
        self.last_robot_position = robot_position[0]

        angle_diff = math.pi - math.fabs(math.fabs(robot_angle[0]-self.last_robot_angle)-math.pi)
        if angle_diff < 0:
            angle_diff = 2*math.pi + angle_diff
        if angle_diff > 0:
            self.cum_heading_changes1 += angle_diff/(current_time-self.last_time)
            self.cum_heading_changes2 += angle_diff
        self.last_robot_angle = robot_angle[0]
        self.last_time = current_time

        min_dist_to_h = 100
        for h in self.people_data:
            dist_to_h = math.sqrt(h.x ** 2 + h.y ** 2)
            if dist_to_h < min_dist_to_h:
                min_dist_to_h = dist_to_h

        if min_dist_to_h < self.minimum_distance_to_human:
            self.minimum_distance_to_human = min_dist_to_h

        if min_dist_to_h <= 0.45:
            self.number_of_intrusions_intimate +=1

        if min_dist_to_h <= 1.2:
            self.number_of_intrusions_personal +=1

        for inter_area in self.interaction_areas:
            if inter_area.contains(Point(0.,0.)):
                self.number_of_intrusions_in_interactions += 1


    #    print("room area", self.room_area, "navigation time", self.navigation_time, "distance", self.distance_traveled, "chc", self.cum_heading_changes,
    #        "minimum distance", self.minimum_distance_to_human,"number of instrusions", self.number_of_intrusions, "number of steps", self.number_of_steps)

    @QtCore.Slot()
    def save_metrics(self):
        time_str = '{0:%Y-%m-%d - %H:%M:%S}'.format(datetime.datetime.now())
        proximity = math.sqrt(self.goal_coordinates[0] ** 2 + self.goal_coordinates[1] ** 2)
        if proximity<=0.5:
            self.goal_reached = True
        else:
            self.goal_reached = False

        metrics = {"date" : time_str,
                    "room_area" : self.room_area,
                    "number_of_humans" : len(self.people_data),
                    "number_of_objects" : len(self.object_data)-1,
                    "number_of_interactions" : len(self.interaction_data),
                    "distance_to_goal"  : self.distance_to_goal,
                    "navigation_time" : self.navigation_time,
                    "distance_traveled" : self.distance_traveled,
                    "chc1" : self.cum_heading_changes1,
                    "chc2" : self.cum_heading_changes2,
                    "minimum_distance_to_human" : self.minimum_distance_to_human,
                    "number_of_intrusions_intimate" : self.number_of_intrusions_intimate,
                    "number_of_intrusions_personal" : self.number_of_intrusions_personal,
                    "number_of_intrusions_in_interactions" : self.number_of_intrusions_in_interactions,
                    "number_of_steps" : self.number_of_steps,
                    "goal_reached": self.goal_reached}
        all_metrics = []
        fname = 'metrics_gaussians.json'
        if os.path.isfile(fname):
            with open(fname, 'r') as fp:
                all_metrics = json.load(fp)

        all_metrics.append(metrics)
        with open(fname, 'w') as fp:    
            json.dump(all_metrics, fp, indent=4)
        print("saving metrics")


    def get_scene(self):

        initial_time = time.time()
        wGrid = 50
        wCell = 8.

        entity_position = dict()
        self.interaction_areas = []


        personsIDs = []
        persons = []
        for h in self.people_data:
            if h.x<4. and h.x>-4. and h.y<4. and h.y>-4. :
                personsIDs.append(h.id)
                p = PersonFields(h.x*1000., h.y*1000., h.angle+math.pi/2.)
                persons.append(p)
                entity_position[h.id] = (h.x, h.y)

        objectsIDs = []
        objects = []
        for o in self.object_data:
            if o.x<4. and o.x>-4. and o.y<4. and o.y>-4. :
                objectsIDs.append(o.id)
                obj = ObjectFields(o.x*1000., o.y*1000., o.angle+math.pi/2., 3.)
                objects.append(obj)
                entity_position[o.id] = (o.x, o.y)

        personGroups = []
        objectInterGroup = []
        for inter in self.interaction_data:                
            newGroup = []
            if inter.idSrc in personsIDs:
                if inter.idDst in personsIDs or inter.idDst in objectsIDs:
                    p1 = entity_position[inter.idSrc]
                    p2 = entity_position[inter.idDst]
                    V = (p2[0] - p1[0], p2[1] - p1[1])
                    lV = math.sqrt(V[0] ** 2 + V[1] ** 2)
                    Vn = (-V[1]/lV, V[0]/lV)
                    inter_area = []
                    inter_area.append((p1[0]-Vn[0]*0.25, p1[1]-Vn[1]*0.25))
                    inter_area.append((p1[0]+Vn[0]*0.25, p1[1]+Vn[1]*0.25))
                    inter_area.append((p2[0]+Vn[0]*0.25, p2[1]+Vn[1]*0.25))
                    inter_area.append((p2[0]-Vn[0]*0.25, p2[1]-Vn[1]*0.25))
                    self.interaction_areas.append(Polygon(inter_area))

                if inter.idDst in personsIDs:
                    personSrc = persons[personsIDs.index(inter.idSrc)]
                    newGroup.append(personSrc)
                    personDst = persons[personsIDs.index(inter.idDst)]
                    newGroup.append(personDst)
                    personGroups.append(newGroup)
                    personsIDs.remove(inter.idSrc)
                    persons.remove(personSrc)
                    personsIDs.remove(inter.idDst)
                    persons.remove(personDst)
                elif inter.idDst in objectsIDs:
                    personSrc = persons[personsIDs.index(inter.idSrc)]
                    objectDst = objects[objectsIDs.index(inter.idDst)]
                    objectDst.angle = personSrc.angle - math.pi
                    objectDst.space = math.sqrt((objectDst.x-personSrc.x) ** 2 + (objectDst.z-personSrc.z) ** 2)/1000. + 0.5
                    objectInterGroup.append((personSrc, objectDst))


             
        for ip in persons:
            personGroups.append([ip])

        next_time1 = time.time()

        scenario = np.full((wGrid, wGrid), 1., np.float32)
        for g in personGroups:
            intimate = []
            personal = []
            social = []
            intimate, personal, social = self.gaussian_spaces.getAllPersonalSpaces(g, False)

            group_scenario = np.full((wGrid, wGrid), 1., np.float32)
            for g in social:
                polyline = []
                for p in g:
                    polyline.append([int(p[0]*wGrid/wCell+wGrid/2), int(-p[1]*wGrid/wCell+wGrid/2)])
                cvpoly = np.array(polyline)
                cv2.fillPoly(group_scenario, [cvpoly], 0.62)
            for g in personal:
                polyline = []
                for p in g:
                    polyline.append([int(p[0]*wGrid/wCell+wGrid/2), int(-p[1]*wGrid/wCell+wGrid/2)])
                cvpoly = np.array(polyline)
                cv2.fillPoly(group_scenario, [cvpoly], 0.53)
            for g in intimate:
                polyline = []
                for p in g:
                    polyline.append([int(p[0]*wGrid/wCell+wGrid/2), int(-p[1]*wGrid/wCell+wGrid/2)])
                cvpoly = np.array(polyline)
                cv2.fillPoly(group_scenario, [cvpoly], 0.)
            scenario = cv2.min(scenario, group_scenario)

        for p, o in objectInterGroup:
            inter_polyline = self.gaussian_spaces.getObjectInteraction([p], [o], True, False)

            group_scenario = np.full((wGrid, wGrid), 1., np.float32)
            for g in inter_polyline:
                polyline = []
                print()
                for p in g:
                    polyline.append([int(p[0]*wGrid/wCell+wGrid/2), int(-p[1]*wGrid/wCell+wGrid/2)])
                cvpoly = np.array(polyline)
                cv2.fillPoly(group_scenario, [cvpoly], 0.62)
            scenario = cv2.min(scenario, group_scenario)

        scenario = cv2.resize(scenario, (100,100), cv2.INTER_CUBIC)

        next_time2 = time.time()

        print("time init", next_time1-initial_time, "time gaussians", next_time2-next_time1)
        #cv2.imshow('scenario', scenario)
        print("number of interactions", len(self.interaction_data))
        return scenario



    def show_image(self):
        self.img_mutex.acquire()
        img = copy.deepcopy(self.img)
        self.img_mutex.release()
        if img is None:
            return
        self.ui.label.setPixmap(QPixmap(QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)))


    #
    # SUBSCRIPTION to gotobjects method from ObjectDetector interface
    #
    def ObjectDetector_gotobjects(self, lst):
        #
        #get robots position
        self.object_data = lst
        for obj in lst:
            if obj.id == -2:
                self.robot_coordinates =  [obj.x, obj.y]
        #
        #pass

    #
    # SUBSCRIPTION to gotinteractions method from InteractionDetector interface
    #
    def InteractionDetector_gotinteractions(self, lst):
        self.interaction_data = lst


    #
    # SUBSCRIPTION to gotpeople method from PeopleDetector interface
    #
    def PeopleDetector_gotpeople(self, lst):
        self.people_data = lst


    #
    # SUBSCRIPTION to gotwalls method from WallDetector interface
    #
    def WallDetector_gotwalls(self, lst):
        self.wall_data = lst


    #
    # SUBSCRIPTION to newsequence method from ByteSequencePublisher interface
    #
    def ByteSequencePublisher_newsequence(self, bs):
        # print('GOT NEW BYTE SEQUENCE', bs)
        # bs = np.load(bs, allow_pickle=True)
        self.img_mutex.acquire()
        self.img = pickle.loads(bs)
        self.img_mutex.release()


    #
    # SUBSCRIPTION to goalupdated method from GoalPublisher interface
    #
    def GoalPublisher_goalupdated(self, goal):
        if self.on_a_mission:
            self.new_goal = True
        self.goal_data = goal
        self.goal_coordinates = [goal.x,  goal.y]
        #pass


    def JoystickAdapter_sendData(self, data):
        if self.on_a_mission:
            self.started_mission = True

        adv = -3.5*data.axes[1].value # m/s
        rot = 4.0*data.axes[0].value  # rad/s
        self.speed_command = [adv, 0, rot]
        self.omnirobot_proxy.setSpeedBase(adv, 0., rot)



    def info(self):
	    dic = {"min_humans":self.configGui.nhumans_min.value(), "max_humans":self.configGui.nhumans_max.value(),
                   "min_wandering_humans":self.configGui.nwandhumans_min.value(), "max_wandering_humans":self.configGui.nwandhumans_max.value()
                   ,"min_plants":self.configGui.nplants_min.value(), "max_plants":self.configGui.nplants_max.value(),
                   "min_tables":self.configGui.ntables_min.value(), "max_tables":self.configGui.ntables_max.value(), 
                   "min_relations":self.configGui.nrelations_min.value(), "max_relations":self.configGui.nrelations_max.value()}
	    self.relations_dic= json.dumps(dic)



