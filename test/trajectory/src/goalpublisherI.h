/*
 *    Copyright (C)2020 by YOUR NAME HERE
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
#ifndef GOALPUBLISHER_H
#define GOALPUBLISHER_H

// Ice includes
#include <Ice/Ice.h>
#include <GoalPublisher.h>

#include <config.h>
#include "genericworker.h"

using namespace RoboCompGoalPublisher;

class GoalPublisherI : public virtual RoboCompGoalPublisher::GoalPublisher
{
public:
GoalPublisherI(GenericWorker *_worker);
	~GoalPublisherI();

	void goalupdated(const GoalT &goal, const Ice::Current&);

private:

	GenericWorker *worker;

};

#endif
