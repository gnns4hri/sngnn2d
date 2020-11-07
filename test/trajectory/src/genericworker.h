/*
 *    Copyright (C) 2020 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef GENERICWORKER_H
#define GENERICWORKER_H

#include "config.h"
#include <stdint.h>
#include <qlog/qlog.h>
#include <CommonBehavior.h>

#include <GenericBase.h>
#include <ObjectDetector.h>
#include <OmniRobot.h>
#include <PeopleDetector.h>
#include <SNGNN2D.h>
#include <WallDetector.h>
#include <GoalPublisher.h>


#define CHECK_PERIOD 5000
#define BASIC_PERIOD 100

using namespace std;
using namespace RoboCompGenericBase;
using namespace RoboCompOmniRobot;
using namespace RoboCompObjectDetector;
using namespace RoboCompPeopleDetector;
using namespace RoboCompWallDetector;
using namespace RoboCompSNGNN2D;
using namespace RoboCompGoalPublisher;

typedef map <string,::IceProxy::Ice::Object*> MapPrx;


class GenericWorker : public QObject
{
Q_OBJECT
public:
	GenericWorker(MapPrx& mprx);
	virtual ~GenericWorker();
	virtual void killYourSelf();
	virtual void setPeriod(int p);

	virtual bool setParams(RoboCompCommonBehavior::ParameterList params) = 0;
	QMutex *mutex;


	RoboCompOmniRobot::OmniRobotPrx omnirobot_proxy;

	virtual void ObjectDetector_gotobjects (const RoboCompObjectDetector::ObjectList &lst) = 0;
	virtual void PeopleDetector_gotpeople (const RoboCompPeopleDetector::PeopleList &lst) = 0;
	virtual void SNGNN2D_gotgrid (const RoboCompSNGNN2D::SNGNN2DData &d) = 0;
	virtual void WallDetector_gotwalls (const RoboCompWallDetector::WallList &lst) = 0;

	virtual void GoalPublisher_goalupdated (const RoboCompGoalPublisher::GoalT &goal) = 0;

protected:

	QTimer timer;
	int Period;

private:


public slots:
	virtual void compute() = 0;
	virtual void initialize(int period) = 0;
	
signals:
	void kill();
};

#endif
