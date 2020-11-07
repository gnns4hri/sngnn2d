#
# Copyright (C) 2016 by YOUR NAME HERE
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

import math
from math import *

import GaussianMix as GM
import checkboundaries as ck
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtCore import QRectF, Qt, QSizeF, QPointF
from PySide2.QtGui import QTransform, QPolygonF
from normal import Normal
from scipy.spatial import ConvexHull


def getPolyline(grid, resolution, lx_inf, ly_inf):
    totalpuntos = []
    for j in range(grid.shape[1]):
        for i in range(grid.shape[0]):
            if grid[j, i] > 0:
                mismocluster, pos = ck.checkboundaries(grid, i, j, totalpuntos)
                if (mismocluster == True):
                    totalpuntos[pos].append([i, j])
                else:
                    puntos = []
                    puntos.append([i, j])
                    totalpuntos.append(puntos)

    ret = []
    for lista in totalpuntos:
        # los puntos en el grid sufren una traslacion, con esto los devolvemos a su posicion original
        for puntos in lista:
            puntos[0] = puntos[0] * resolution + lx_inf
            puntos[1] = puntos[1] * resolution + ly_inf

        points = np.asarray(lista)
        ##ConcaveHull --> Mas puntos pero la forma se aproxima mas
        # hull = CH.concaveHull(points,3)
        # ret.append(hull)
        # ## ConvexHull --> Menos puntos pero la forma es el contorno
        hull = ConvexHull(points)
        ret.append(points[hull.vertices])

    return ret
class SNGPoint2D():
    def __init__(self, x=0, z=0):
        self.x = x
        self.z = z    

class Person(object):
    x = 0
    y = 0
    th = 0
    vel = 0

    _radius = 0.30

    """ Public Methods """

    def __init__(self, x=0, y=0, th=0, vel=0):
        self.x = x / 1000
        self.y = y / 1000
        self.th = th
        self.vel = vel

    def draw(self, sigma_h, sigma_r, sigma_s, rot, drawPersonalSpace=False):
        # define grid.
        npts = 50
        x = np.linspace(self.x - 4, self.x + 4, npts)
        y = np.linspace(self.y - 4, self.y + 4, npts)

        X, Y = np.meshgrid(x, y)
        Z = self._calculatePersonalSpace(X, Y, sigma_h, sigma_r, sigma_s, rot)

        if drawPersonalSpace:
            plt.contour(X, Y, Z, 10)

            # Corpo
            body = plt.Circle((self.x, self.y), radius=self._radius, fill=False)
            plt.gca().add_patch(body)

            # Orientacion
            # print("alpha:")
            x_aux = self.x + self._radius * cos(pi / 2 - self.th);
            # print(self.x)
            # print(x_aux);
            y_aux = self.y + self._radius * sin(pi / 2 - self.th);
            # print(self.y)

            # print(y_aux)
            heading = plt.Line2D((self.x, x_aux), (self.y, y_aux), lw=1, color='k')
            plt.gca().add_line(heading)

            plt.axis('equal')

        return Z

    """ Private Methods """

    def _calculatePersonalSpace(self, x, y, sigma_h, sigma_r, sigma_s, rot):
        alpha = np.arctan2(y - self.y, x - self.x) - rot - pi / 2
        nalpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalizando no intervalo [-pi, pi)

        sigma = np.copy(nalpha)
        for i in range(nalpha.shape[0]):
            for j in range(nalpha.shape[1]):
                sigma[i, j] = sigma_r if nalpha[i, j] <= 0 else sigma_h

        a = cos(rot) ** 2 / 2 * sigma ** 2 + sin(rot) ** 2 / 2 * sigma_s ** 2
        b = sin(2 * rot) / 4 * sigma ** 2 - sin(2 * rot) / 4 * sigma_s ** 2
        c = sin(rot) ** 2 / 2 * sigma ** 2 + cos(rot) ** 2 / 2 * sigma_s ** 2

        z = np.exp(-(a * (x - self.x) ** 2 + 2 * b * (x - self.x) * (y - self.y) + c * (y - self.y) ** 2))

        return z


class Object():
    def __init__(self, x=0, y=0, th=0, sp=0):
        self.x = x
        self.y = y
        self.th = th
        self.sp = sp


class GaussianSpaces():
    def __init__(self):
        pass

    # plt.ion()

    #
    # getAllPersonalSpaces
    #
    def getAllPersonalSpaces(self, persons, represent):

        personal_spaces = ["intimate", "personal", "social"]
                                        # sigma_h, sigma_r, sigma_s,  h
        dict_space_param = {"intimate": [1.3,       1.,     1.3,    0.8],
                            "personal": [1.3,       1.,     1.3,    0.4],
                            "social":   [3.,        1.,     1.3,    0.1],
                            }

        dict_space_polylines = {"intimate": [],
                                "personal": [],
                                "social": [],
                                }

        dict_spaces_to_plot = {"intimate": [],
                               "personal": [],
                               "social": [],
                               }

        ##Limites de la representacion
        lx_inf = -6
        lx_sup = 10
        ly_inf = -6
        ly_sup = 10

        for space in personal_spaces:
            normals = []
            for p in persons:
                person = Person(p.x, p.z, p.angle)
                # print('Pose x', person.x, 'Pose z', person.y, 'Rotacion', person.th)
                # person.draw(2,1, 4./3.,pi/2 - person.th, drawPersonalSpace=dibujar) #Valores originales
                person.draw(dict_space_param[space][0], dict_space_param[space][1], dict_space_param[space][2],
                            pi / 2 - person.th, drawPersonalSpace=represent)
                normals.append(Normal(mu=[[person.x], [person.y]],
                                      sigma=[-person.th - pi / 2., dict_space_param[space][0],
                                             dict_space_param[space][1],
                                             dict_space_param[space][2]], elliptical=True))
            # print ("numero de gaussianas",len(normals))

            resolution = 0.1
            limits = [[lx_inf, lx_sup], [ly_inf, ly_sup]]
            _, z = Normal.makeGrid(normals, dict_space_param[space][3], 2, limits=limits, resolution=resolution)
            grid = GM.filterEdges(z, dict_space_param[space][3])

            totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

            for pol in totalpuntosorden:
                polyline = []
                polyline_to_plt = []

                for pnt in pol:
                    punto = SNGPoint2D()
                    punto.x = pnt[0] * 1000
                    punto.z = pnt[1] * 1000
                    polyline.append(punto)

                    polyline_to_plt.append([pnt[0], pnt[1]])

                dict_space_polylines[space].append(polyline_to_plt)

                if len(polyline_to_plt) != 0:
                    dict_spaces_to_plot[space].append(polyline_to_plt)

        if represent:
            for soc in dict_spaces_to_plot["social"]:
                x, y = zip(*soc)
                plt.plot(x, y, color='c', marker='.')

            for per in dict_spaces_to_plot["personal"]:
                x, y = zip(*per)
                plt.plot(x, y, color='m', marker='.')

            for inti in dict_spaces_to_plot["intimate"]:
                x, y = zip(*inti)
                plt.plot(x, y, color='r', marker='.')

            plt.axis('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        return dict_space_polylines['intimate'], dict_space_polylines['personal'], dict_space_polylines['social']

    #
    # getPersonalSPace
    #
    def SocialNavigationGaussian_getPersonalSpace(self, persons, h, dibujar):

        plt.close('all')

        ##Limites de la representacion
        lx_inf = -6
        lx_sup = 10
        ly_inf = -6
        ly_sup = 10

        ##########################################CLUSTERING##################################################

        normals = []

        for p in persons:
            pn = Person(p.x, p.z, p.angle)
            # print('Pose x', pn.x, 'Pose z', pn.y, 'Rotacion', pn.th)
            # pn.draw(2,1, 4./3.,pi/2 - pn.th, drawPersonalSpace=dibujar) #Valores originales
            pn.draw(1.3, 1., 1.3, pi / 2 - pn.th, drawPersonalSpace=dibujar)
            normals.append(Normal(mu=[[pn.x], [pn.y]], sigma=[-pn.th - pi / 2., 1.3, 1., 1.3], elliptical=True))
            # normals.append(Normal(mu=[[pn.x], [pn.y]], sigma=[-pn.th - pi/2., 2, 1, 4. / 3], elliptical=True))
        # print ("numero de gaussianas",len(normals))

        resolution = 0.1
        limits = [[lx_inf, lx_sup], [ly_inf, ly_sup]]
        _, z = Normal.makeGrid(normals, h, 2, limits=limits, resolution=resolution)
        grid = GM.filterEdges(z, h)

        ###########################LEO EL GRID Y SEPARO LAS POLILINEAS, DESPUES SE HACE CONVEXHULL####################################
        polylines = []
        totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

        for pol in totalpuntosorden:
            polyline = []
            for pnt in pol:
                punto = SNGPoint2D()
                punto.x = pnt[0] * 1000
                punto.z = pnt[1] * 1000
                polyline.append(punto)
            polylines.append(polyline)

        if (dibujar):

            ###DIBUJO ZONA Personal
            _, z = Normal.makeGrid(normals, 0.4, 2, limits=limits, resolution=resolution)
            grid = GM.filterEdges(z, 0.4)

            polylines = []
            totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

            for pol in totalpuntosorden:
                polyline = []
                for pnt in pol:
                    punto = SNGPoint2D()
                    punto.x = pnt[0]
                    punto.z = pnt[1]
                    polyline.append(punto)
                polylines.append(polyline)

            for ps in polylines:
                # plt.figure()
                for p in ps:
                    plt.plot(p.x, p.z, "om-")
                    plt.axis('equal')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                # plt.show()

            ###DIBUJO ZONA INTIMA
            _, z = Normal.makeGrid(normals, 0.8, 2, limits=limits, resolution=resolution)
            grid = GM.filterEdges(z, 0.8)

            polylines = []
            totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

            for pol in totalpuntosorden:
                polyline = []
                for pnt in pol:
                    punto = SNGPoint2D()
                    punto.x = pnt[0]
                    punto.z = pnt[1]
                    polyline.append(punto)
                polylines.append(polyline)

            for ps in polylines:
                # plt.figure()
                for p in ps:
                    plt.plot(p.x, p.z, "or-")
                    plt.axis('equal')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                # plt.show()

        plt.show()
        return polylines

    #
    # getSocialSpace
    #
    def SocialNavigationGaussian_getSocialSpace(self, persons, h, draw):
        plt.close('all')

        ##Limites de la representacion

        lx_inf = -6
        lx_sup = 10
        ly_inf = -6
        ly_sup = 10

        ##########################################CLUSTERING##################################################

        normals = []

        for p in persons:
            pn = Person(p.x / 1000, p.z / 1000, p.angle)
            pn.draw(3, 1, 1.3, pi / 2 - pn.th, drawPersonalSpace=draw)  # Valores originales
            normals.append(Normal(mu=[[pn.x], [pn.y]], sigma=[-pn.th - pi / 2., 3, 1, 1.3], elliptical=True))
        # print ("numero de gaussianas",len(normals))

        resolution = 0.1
        limits = [[lx_inf, lx_sup], [ly_inf, ly_sup]]
        _, z = Normal.makeGrid(normals, h, 2, limits=limits, resolution=resolution)
        grid = GM.filterEdges(z, h)

        ###########################LEO EL GRID Y SEPARO LAS POLILINEAS, DESPUES SE HACE CONVEXHULL####################################
        polylines = []
        totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

        for pol in totalpuntosorden:
            polyline = []
            for pnt in pol:
                punto = SNGPoint2D()
                punto.x = pnt[0] * 1000
                punto.z = pnt[1] * 1000
                polyline.append(punto)
            polylines.append(polyline)

        if (draw):
            ##DIBUJO ZONA Social
            _, z = Normal.makeGrid(normals, 0.1, 2, limits=limits, resolution=resolution)
            grid = GM.filterEdges(z, 0.1)

            polylines = []
            totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

            for pol in totalpuntosorden:
                polyline = []
                for pnt in pol:
                    punto = SNGPoint2D()
                    punto.x = pnt[0]
                    punto.z = pnt[1]
                    polyline.append(punto)
                polylines.append(polyline)

            for ps in polylines:
                # plt.figure()
                for p in ps:
                    plt.plot(p.x, p.z, "oc-")
                    plt.axis('equal')
                    plt.xlabel('X')
                    plt.ylabel('Y')

        plt.show()
        return polylines

    def SocialNavigationGaussian_getPassOnRight(self, persons, h, dibujar):

        plt.close("all")

        lx_inf = -5
        lx_sup = 5
        ly_inf = -5
        ly_sup = 5

        normals = []

        for p in persons:
            pn = Person(p.x / 1000, p.z / 1000, p.angle, p.vel)
            pn.draw(2., 1, 4. / 3., pi / 2 - pn.th, drawPersonalSpace=dibujar)
            pn.draw(2., 1, 4. / 3., pi - pn.th, drawPersonalSpace=dibujar)
            normals.append(Normal(mu=[[pn.x], [pn.y]], sigma=[-pn.th - pi / 2., 2, 1, 4. / 3], elliptical=True))
            normals.append(Normal(mu=[[pn.x], [pn.y]], sigma=[-pn.th, 2, 0.75, 4. / 3], elliptical=True))
        # h = 0.4
        # h = prox / 100
        resolution = 0.1
        limits = [[lx_inf, lx_sup], [ly_inf, ly_sup]]
        _, z = Normal.makeGrid(normals, h, 2, limits=limits,
                               resolution=resolution)  # Las posiciones de las personas tienen que estar en metros
        grid = GM.filterEdges(z, h)

        if (dibujar):
            plt.figure()
            plt.imshow(grid, extent=[lx_inf, lx_sup, ly_inf, ly_sup], shape=grid.shape, interpolation='none',
                       aspect='equal', origin='lower', cmap='Greys', vmin=0, vmax=2)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')

        np.savetxt('log.txt', grid, fmt='%i')

        polylines = []
        totalpuntosorden = getPolyline(grid, resolution, lx_inf, ly_inf)

        for pol in totalpuntosorden:
            polyline = []
            for pnt in pol:
                punto = SNGPoint2D()
                punto.x = pnt[0] * 1000
                punto.z = pnt[1] * 1000
                polyline.append(punto)
            polylines.append(polyline)
        plt.show()
        return polylines

    #
    # getObjectInteraction
    #
    def getObjectInteraction(self, persons, objects, interaction, d):

        # print("getObjectInteration")
        plt.close('all')

        polylines_object = []
        polylines_interacting = []

        for o in objects:
            obj = Object(o.x/1000., o.z/1000., o.angle, o.space)
            # print("OBJETO")
            ##para dibujarlo
            if d:
                plt.figure('ObjectSpace')
                rect = plt.Rectangle((obj.x - 0.25, obj.y - 0.25), 0.5, 0.5, fill=False)

                plt.gca().add_patch(rect)
                x_aux = obj.x + 0.25 * cos(pi / 2 - obj.th)
                y_aux = obj.y + 0.25 * sin(pi / 2 - obj.th)
                heading = plt.Line2D((obj.x, x_aux), (obj.y, y_aux), lw=1, color='k')
                plt.gca().add_line(heading)

            w = 1.0
            # print (obj.x,obj.y)
            ##para calcular el rectangulo
            s = QRectF(QPointF(0, 0), QSizeF(w, obj.sp))

            # if (d):
            #     plt.plot (s.bottomLeft().x(),s.bottomLeft().y(),"go")
            #     plt.plot(s.bottomRight().x(), s.bottomRight().y(), "ro")
            #     plt.plot(s.topRight().x(), s.topRight().y(), "yo")
            #     plt.plot(s.topLeft().x(), s.topLeft().y(), "bo")

            space = QPolygonF()
            space.append(s.topLeft())
            space.append(s.topRight())
            space.append(QPointF(s.bottomRight().x() + obj.sp / 4, s.bottomRight().y()))
            space.append(QPointF(s.bottomLeft().x() - obj.sp / 4, s.bottomLeft().y()))

            t = QTransform()
            t.translate(-w / 2, 0)
            space = t.map(space)
            t = QTransform()
            t.rotateRadians(-obj.th)
            space = t.map(space)

            t = QTransform()
            t.translate(obj.x, obj.y)
            space = t.map(space)

            # points = []
            # for x in xrange(space.count()-1):
            #     point = space.value(x)
            #     print ("valor", point)
            #     points.append([point.x(),point.y()])
            #     plt.plot(point.x(),point.y(),"go")

            polyline = []

            for x in range(space.count()):
                point = space.value(x)
                if (d):
                    plt.plot(point.x(), point.y(), "go")

                p = SNGPoint2D()
                p.x = point.x()
                p.z = point.y()
                polyline.append([p.x, p.z])

            polylines_object.append(polyline)

            for p in persons:
                pn = Person(p.x, p.z, p.angle)
                # print("PERSONA", persons.index(p)+1)
                if d:
                    body = plt.Circle((pn.x, pn.y), radius=0.3, fill=False)
                    plt.gca().add_patch(body)

                    x_aux = pn.x + 0.30 * cos(pi / 2 - pn.th)
                    y_aux = pn.y + 0.30 * sin(pi / 2 - pn.th)
                    heading = plt.Line2D((pn.x, x_aux), (pn.y, y_aux), lw=1, color='k')
                    plt.gca().add_line(heading)
                    plt.axis('equal')

                ##CHECKING THE ORIENTATION
                print("obj.angle", obj.th, "person.angle", pn.th)
                a = abs(obj.th - abs(pn.th - math.pi))
                if a < math.radians(45):
                    checkangle = True
                else:
                    checkangle = False

                ##CHECKING IF THE PERSON IS INSIDE THE POLYGON
                if space.containsPoint(QPointF(pn.x, pn.y), Qt.OddEvenFill):# and checkangle:
                    print("DENTROOOOO Y MIRANDO")
                    if not polyline in polylines_interacting:
                        polylines_interacting.append(polyline)

        if d:
            for ps in polylines_interacting:
                #  plt.figure()
                for p in ps:
                    plt.plot(p.x, p.z, "ro")
                    plt.axis('equal')
                    plt.xlabel('X')
                    plt.ylabel('Y')
            plt.show()
        plt.show()

        if (interaction):
            return polylines_interacting
        else:
            return polylines_object
