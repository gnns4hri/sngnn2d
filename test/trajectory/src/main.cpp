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


/** \mainpage RoboComp::trajectory
 *
 * \section intro_sec Introduction
 *
 * The trajectory component...
 *
 * \section interface_sec Interface
 *
 * interface...
 *
 * \section install_sec Installation
 *
 * \subsection install1_ssec Software depencences
 * ...
 *
 * \subsection install2_ssec Compile and install
 * cd trajectory
 * <br>
 * cmake . && make
 * <br>
 * To install:
 * <br>
 * sudo make install
 *
 * \section guide_sec User guide
 *
 * \subsection config_ssec Configuration file
 *
 * <p>
 * The configuration file etc/config...
 * </p>
 *
 * \subsection execution_ssec Execution
 *
 * Just: "${PATH_TO_BINARY}/trajectory --Ice.Config=${PATH_TO_CONFIG_FILE}"
 *
 * \subsection running_ssec Once running
 *
 * ...
 *
 */
#include <signal.h>

// QT includes
#include <QtCore>
#include <QtGui>

// ICE includes
#include <Ice/Ice.h>
#include <IceStorm/IceStorm.h>
#include <Ice/Application.h>

#include <rapplication/rapplication.h>
#include <sigwatch/sigwatch.h>
#include <qlog/qlog.h>

#include "config.h"
#include "genericmonitor.h"
#include "genericworker.h"
#include "specificworker.h"
#include "specificmonitor.h"
#include "commonbehaviorI.h"

#include <objectdetectorI.h>
#include <peopledetectorI.h>
#include <sngnn2dI.h>
#include <walldetectorI.h>
#include <goalpublisherI.h>

#include <GenericBase.h>


// User includes here

// Namespaces
using namespace std;
using namespace RoboCompCommonBehavior;

class trajectory : public RoboComp::Application
{
public:
	trajectory (QString prfx) { prefix = prfx.toStdString(); }
private:
	void initialize();
	std::string prefix;
	MapPrx mprx;

public:
	virtual int run(int, char*[]);
};

void ::trajectory::initialize()
{
	// Config file properties read example
	// configGetString( PROPERTY_NAME_1, property1_holder, PROPERTY_1_DEFAULT_VALUE );
	// configGetInt( PROPERTY_NAME_2, property1_holder, PROPERTY_2_DEFAULT_VALUE );
}

int ::trajectory::run(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);  // NON-GUI application


	sigset_t sigs;
	sigemptyset(&sigs);
	sigaddset(&sigs, SIGHUP);
	sigaddset(&sigs, SIGINT);
	sigaddset(&sigs, SIGTERM);
	sigprocmask(SIG_UNBLOCK, &sigs, 0);

	UnixSignalWatcher sigwatch;
	sigwatch.watchForSignal(SIGINT);
	sigwatch.watchForSignal(SIGTERM);
	QObject::connect(&sigwatch, SIGNAL(unixSignal(int)), &a, SLOT(quit()));

	int status=EXIT_SUCCESS;

	OmniRobotPrx omnirobot_proxy;

	string proxy, tmp;
	initialize();


	try
	{
		if (not GenericMonitor::configGetString(communicator(), prefix, "OmniRobotProxy", proxy, ""))
		{
			cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy OmniRobotProxy\n";
		}
		omnirobot_proxy = OmniRobotPrx::uncheckedCast( communicator()->stringToProxy( proxy ) );
	}
	catch(const Ice::Exception& ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception creating proxy OmniRobot: " << ex;
		return EXIT_FAILURE;
	}
	rInfo("OmniRobotProxy initialized Ok!");

	mprx["OmniRobotProxy"] = (::IceProxy::Ice::Object*)(&omnirobot_proxy);//Remote server proxy creation example
	IceStorm::TopicManagerPrx topicManager;
	try
	{
		topicManager = IceStorm::TopicManagerPrx::checkedCast(communicator()->propertyToProxy("TopicManager.Proxy"));
	}
	catch (const Ice::Exception &ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception: STORM not running: " << ex << endl;
		return EXIT_FAILURE;
	}

	SpecificWorker *worker = new SpecificWorker(mprx);
	//Monitor thread
	SpecificMonitor *monitor = new SpecificMonitor(worker,communicator());
	QObject::connect(monitor, SIGNAL(kill()), &a, SLOT(quit()));
	QObject::connect(worker, SIGNAL(kill()), &a, SLOT(quit()));
	monitor->start();

	if ( !monitor->isRunning() )
		return status;

	while (!monitor->ready)
	{
		usleep(10000);
	}

	try
	{
		try {
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "CommonBehavior.Endpoints", tmp, "")) {
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy CommonBehavior\n";
			}
			Ice::ObjectAdapterPtr adapterCommonBehavior = communicator()->createObjectAdapterWithEndpoints("commonbehavior", tmp);
			CommonBehaviorI *commonbehaviorI = new CommonBehaviorI(monitor);
			adapterCommonBehavior->add(commonbehaviorI, Ice::stringToIdentity("commonbehavior"));
			adapterCommonBehavior->activate();
		}
		catch(const Ice::Exception& ex)
		{
			status = EXIT_FAILURE;

			cout << "[" << PROGRAM_NAME << "]: Exception raised while creating CommonBehavior adapter: " << endl;
			cout << ex;

		}




		// Server adapter creation and publication
		IceStorm::TopicPrx objectdetector_topic;
		Ice::ObjectPrx objectdetector;
		try
		{
			if (not GenericMonitor::configGetString(communicator(), prefix, "ObjectDetectorTopic.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy ObjectDetectorProxy";
			}
			Ice::ObjectAdapterPtr ObjectDetector_adapter = communicator()->createObjectAdapterWithEndpoints("objectdetector", tmp);
			ObjectDetectorPtr objectdetectorI_ =  new ObjectDetectorI(worker);
			Ice::ObjectPrx objectdetector = ObjectDetector_adapter->addWithUUID(objectdetectorI_)->ice_oneway();
			if(!objectdetector_topic)
			{
				try {
					objectdetector_topic = topicManager->create("ObjectDetector");
				}
				catch (const IceStorm::TopicExists&) {
					//Another client created the topic
					try{
						cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
						objectdetector_topic = topicManager->retrieve("ObjectDetector");
					}
					catch(const IceStorm::NoSuchTopic&)
					{
						cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
						//Error. Topic does not exist
					}
				}
				catch(const IceUtil::NullHandleException&)
				{
					cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
					"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
					return EXIT_FAILURE;
				}
				IceStorm::QoS qos;
				objectdetector_topic->subscribeAndGetPublisher(qos, objectdetector);
			}
			ObjectDetector_adapter->activate();
		}
		catch(const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Error creating ObjectDetector topic.\n";
			//Error. Topic does not exist
		}

		// Server adapter creation and publication
		IceStorm::TopicPrx peopledetector_topic;
		Ice::ObjectPrx peopledetector;
		try
		{
			if (not GenericMonitor::configGetString(communicator(), prefix, "PeopleDetectorTopic.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy PeopleDetectorProxy";
			}
			Ice::ObjectAdapterPtr PeopleDetector_adapter = communicator()->createObjectAdapterWithEndpoints("peopledetector", tmp);
			PeopleDetectorPtr peopledetectorI_ =  new PeopleDetectorI(worker);
			Ice::ObjectPrx peopledetector = PeopleDetector_adapter->addWithUUID(peopledetectorI_)->ice_oneway();
			if(!peopledetector_topic)
			{
				try {
					peopledetector_topic = topicManager->create("PeopleDetector");
				}
				catch (const IceStorm::TopicExists&) {
					//Another client created the topic
					try{
						cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
						peopledetector_topic = topicManager->retrieve("PeopleDetector");
					}
					catch(const IceStorm::NoSuchTopic&)
					{
						cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
						//Error. Topic does not exist
					}
				}
				catch(const IceUtil::NullHandleException&)
				{
					cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
					"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
					return EXIT_FAILURE;
				}
				IceStorm::QoS qos;
				peopledetector_topic->subscribeAndGetPublisher(qos, peopledetector);
			}
			PeopleDetector_adapter->activate();
		}
		catch(const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Error creating PeopleDetector topic.\n";
			//Error. Topic does not exist
		}

		// Server adapter creation and publication
		IceStorm::TopicPrx sngnn2d_topic;
		Ice::ObjectPrx sngnn2d;
		try
		{
			if (not GenericMonitor::configGetString(communicator(), prefix, "SNGNN2DTopic.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy SNGNN2DProxy";
			}
			Ice::ObjectAdapterPtr SNGNN2D_adapter = communicator()->createObjectAdapterWithEndpoints("sngnn2d", tmp);
			SNGNN2DPtr sngnn2dI_ =  new SNGNN2DI(worker);
			Ice::ObjectPrx sngnn2d = SNGNN2D_adapter->addWithUUID(sngnn2dI_)->ice_oneway();
			if(!sngnn2d_topic)
			{
				try {
					sngnn2d_topic = topicManager->create("SNGNN2D");
				}
				catch (const IceStorm::TopicExists&) {
					//Another client created the topic
					try{
						cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
						sngnn2d_topic = topicManager->retrieve("SNGNN2D");
					}
					catch(const IceStorm::NoSuchTopic&)
					{
						cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
						//Error. Topic does not exist
					}
				}
				catch(const IceUtil::NullHandleException&)
				{
					cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
					"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
					return EXIT_FAILURE;
				}
				IceStorm::QoS qos;
				sngnn2d_topic->subscribeAndGetPublisher(qos, sngnn2d);
			}
			SNGNN2D_adapter->activate();
		}
		catch(const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Error creating SNGNN2D topic.\n";
			//Error. Topic does not exist
		}

        // Server adapter creation and publication
        IceStorm::TopicPrx walldetector_topic;
        Ice::ObjectPrx walldetector;
        try
        {
            if (not GenericMonitor::configGetString(communicator(), prefix, "WallDetectorTopic.Endpoints", tmp, ""))
            {
                cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy WallDetectorProxy";
            }
            Ice::ObjectAdapterPtr WallDetector_adapter = communicator()->createObjectAdapterWithEndpoints("walldetector", tmp);
            RoboCompWallDetector::WallDetectorPtr walldetectorI_ =  new WallDetectorI(worker);
            Ice::ObjectPrx walldetector = WallDetector_adapter->addWithUUID(walldetectorI_)->ice_oneway();
            if(!walldetector_topic)
            {
                try {
                    walldetector_topic = topicManager->create("WallDetector");
                }
                catch (const IceStorm::TopicExists&) {
                    //Another client created the topic
                    try{
                        cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
                        walldetector_topic = topicManager->retrieve("WallDetector");
                    }
                    catch(const IceStorm::NoSuchTopic&)
                    {
                        cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
                        //Error. Topic does not exist
                    }
                }
                catch(const IceUtil::NullHandleException&)
                {
                    cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
                    "\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
                    return EXIT_FAILURE;
                }
                IceStorm::QoS qos;
                walldetector_topic->subscribeAndGetPublisher(qos, walldetector);
            }
            WallDetector_adapter->activate();
        }
        catch(const IceStorm::NoSuchTopic&)
        {
            cout << "[" << PROGRAM_NAME << "]: Error creating WallDetector topic.\n";
            //Error. Topic does not exist
        }

        // Server adapter creation and publication
        IceStorm::TopicPrx goalpublisher_topic;
        Ice::ObjectPrx goalpublisher;
        try
        {
            if (not GenericMonitor::configGetString(communicator(), prefix, "GoalPublisherTopic.Endpoints", tmp, ""))
            {
                cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy GoalPublisherProxy";
            }
            Ice::ObjectAdapterPtr GoalPublisher_adapter = communicator()->createObjectAdapterWithEndpoints("goalpublisher", tmp);
            RoboCompGoalPublisher::GoalPublisherPtr goalpublisherI_ =  new GoalPublisherI(worker);
            Ice::ObjectPrx goalpublisher = GoalPublisher_adapter->addWithUUID(goalpublisherI_)->ice_oneway();
            if(!goalpublisher_topic)
            {
                try {
                    goalpublisher_topic = topicManager->create("GoalPublisher");
                }
                catch (const IceStorm::TopicExists&) {
                    //Another client created the topic
                    try{
                        cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
                        goalpublisher_topic = topicManager->retrieve("GoalPublisher");
                    }
                    catch(const IceStorm::NoSuchTopic&)
                    {
                        cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
                        //Error. Topic does not exist
                    }
                }
                catch(const IceUtil::NullHandleException&)
                {
                    cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
                    "\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
                    return EXIT_FAILURE;
                }
                IceStorm::QoS qos;
                goalpublisher_topic->subscribeAndGetPublisher(qos, goalpublisher);
            }
            GoalPublisher_adapter->activate();
        }
        catch(const IceStorm::NoSuchTopic&)
        {
            cout << "[" << PROGRAM_NAME << "]: Error creating GoalPublisher topic.\n";
            //Error. Topic does not exist
        }

		// Server adapter creation and publication
		cout << SERVER_FULL_NAME " started" << endl;

		// User defined QtGui elements ( main window, dialogs, etc )

		#ifdef USE_QTGUI
			//ignoreInterrupt(); // Uncomment if you want the component to ignore console SIGINT signal (ctrl+c).
			a.setQuitOnLastWindowClosed( true );
		#endif
		// Run QT Application Event Loop
		a.exec();

		try
		{
			std::cout << "Unsubscribing topic: objectdetector " <<std::endl;
			objectdetector_topic->unsubscribe( objectdetector );
		}
		catch(const Ice::Exception& ex)
		{
			std::cout << "ERROR Unsubscribing topic: objectdetector " <<std::endl;
		}
		try
		{
			std::cout << "Unsubscribing topic: peopledetector " <<std::endl;
			peopledetector_topic->unsubscribe( peopledetector );
		}
		catch(const Ice::Exception& ex)
		{
			std::cout << "ERROR Unsubscribing topic: peopledetector " <<std::endl;
		}
		try
		{
			std::cout << "Unsubscribing topic: sngnn2d " <<std::endl;
			sngnn2d_topic->unsubscribe( sngnn2d );
		}
		catch(const Ice::Exception& ex)
		{
			std::cout << "ERROR Unsubscribing topic: sngnn2d " <<std::endl;
		}

		status = EXIT_SUCCESS;
	}
	catch(const Ice::Exception& ex)
	{
		status = EXIT_FAILURE;

		cout << "[" << PROGRAM_NAME << "]: Exception raised on main thread: " << endl;
		cout << ex;

	}
	#ifdef USE_QTGUI
		a.quit();
	#endif

	status = EXIT_SUCCESS;
	monitor->terminate();
	monitor->wait();
	delete worker;
	delete monitor;
	return status;
}

int main(int argc, char* argv[])
{
	string arg;

	// Set config file
	std::string configFile = "config";
	if (argc > 1)
	{
		std::string initIC("--Ice.Config=");
		size_t pos = std::string(argv[1]).find(initIC);
		if (pos == 0)
		{
			configFile = std::string(argv[1]+initIC.size());
		}
		else
		{
			configFile = std::string(argv[1]);
		}
	}

	// Search in argument list for --prefix= argument (if exist)
	QString prefix("");
	QString prfx = QString("--prefix=");
	for (int i = 2; i < argc; ++i)
	{
		arg = argv[i];
		if (arg.find(prfx.toStdString(), 0) == 0)
		{
			prefix = QString::fromStdString(arg).remove(0, prfx.size());
			if (prefix.size()>0)
				prefix += QString(".");
			printf("Configuration prefix: <%s>\n", prefix.toStdString().c_str());
		}
	}
	::trajectory app(prefix);

	return app.main(argc, argv, configFile.c_str());
}
