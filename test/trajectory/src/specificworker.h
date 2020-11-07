/*
 *	Copyright (C)2020 by YOUR NAME HERE
 *
 *	RoboComp is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	RoboComp is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	   \brief
	   @author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <mutex>


#include <genericworker.h>
#include <innermodel/innermodel.h>
#include <time.h>

#include <opencv2/opencv.hpp>

#define SOCNAV_AREA_WIDTH 800.
#define GRID_SIZE1 50
#define GRID_SIZE2 200
#define DRAW_SIZE 500
#define GRID_BORDER 12


using namespace cv;

struct nodeType
{
    cv::Point p;
    float cost;
};

struct wallType
{
    cv::Point2f p1, p2;
};


typedef std::vector<Point2f> BBType;

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(MapPrx& mprx);
    ~SpecificWorker();
    bool setParams(RoboCompCommonBehavior::ParameterList params);

    void ObjectDetector_gotobjects(const ObjectList &lst);
    void PeopleDetector_gotpeople(const PeopleList &lst);
    void SNGNN2D_gotgrid(const SNGNN2DData &d);
    void WallDetector_gotwalls(const RoboCompWallDetector::WallList &lst);
    void GoalPublisher_goalupdated(const RoboCompGoalPublisher::GoalT &goal);

public slots:
    void compute();
    void initialize(int period);

private:

    std::mutex mtx_read, mtx_write, mtx_objects, mtx_walls, mtx_goal;
    SNGNN2DData data_read, data_write;


    float target_x;
    float target_y;
    float world_target_x;
    float world_target_y;

    float robot_x;
    float robot_y;
    float robot_angle;

    std::vector<BBType> objectBB_list;
    std::vector<wallType> wall_list;

    Matx33f w2r, r2w;

    cv::Mat grid;
    cv::Mat grid_big;
    cv::Mat visible_grid;

    float prev_rot, prev_adv;

    std::vector<Point> neighList;

    std::vector<Point> final_path, world_path;

    std::vector<Point> find_free_path(Mat grid, Point start, Point goal);
    bool check_free_path(std::vector<Point> path, Mat grid);
    std::vector<Point> create_final_path(std::vector<Point>, float s=1.);
    float grid_cost(Mat grid, Point p);
    std::vector<nodeType> validActions(Mat grid, Point p, int grid_border);
    float h_function(Point p1, Point p2);
    std::vector<Point> path_planning(Mat grid, Point start, Point goal, int grid_border);
    void follow_path(std::vector<Point> path);

    void initialiseNeighbours();
    void add_objects_to_grid(Mat& grid);
    void add_walls_to_grid(Mat& grid);
    float compute_area_cost(Mat& _grid, Point center, int w);
    bool check_free_area(Mat& grid, Point center, int w, float limit);
    void redefine_target(int gx, int gy, int & new_gx, int & new_gy);
};

#endif
