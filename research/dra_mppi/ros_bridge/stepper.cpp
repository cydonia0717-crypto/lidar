// Reproduction glue. The original simulator/ROS conversion implementation is
// compiled by reference; its model and prediction functions are not rewritten.
#define main upstream_original_ros_main
#include DRA_UPSTREAM_ROS_CPP
#undef main
#include <dra_repro_bridge/Step.h>
#include <ros/callback_queue.h>
#include <boost/make_shared.hpp>
#include <boost/function.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "dra_social_stepper");
  ros::NodeHandle nh;
  ROSPedestrianSimulator simulator;
  mpc_planner_msgs::ObstacleArray predictions;
  derived_object_msgs::ObjectArray observations;
  unsigned int prediction_seq = 0, observation_seq = 0;
  auto prediction_sub = nh.subscribe<mpc_planner_msgs::ObstacleArray>(
      "/pedestrian_simulator/trajectory_predictions", 1,
      [&](const mpc_planner_msgs::ObstacleArray::ConstPtr& msg) { predictions = *msg; ++prediction_seq; });
  auto observation_sub = nh.subscribe<derived_object_msgs::ObjectArray>(
      "/pedestrian_simulator/pedestrians", 1,
      [&](const derived_object_msgs::ObjectArray::ConstPtr& msg) { observations = *msg; ++observation_seq; });
  boost::function<bool(dra_repro_bridge::Step::Request&, dra_repro_bridge::Step::Response&)> callback =
      [&](dra_repro_bridge::Step::Request& req, dra_repro_bridge::Step::Response& res) {
        const auto p0 = prediction_seq, o0 = observation_seq;
        simulator.RobotStateCallback(boost::make_shared<geometry_msgs::PoseStamped>(req.robot_state));
        simulator.Loop(ros::TimerEvent());
        const auto deadline = ros::WallTime::now() + ros::WallDuration(2.0);
        while (ros::ok() && (prediction_seq == p0 || observation_seq == o0) && ros::WallTime::now() < deadline)
          ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.005));
        res.fresh = prediction_seq > p0 && observation_seq > o0;
        res.predictions = predictions;
        res.observations = observations;
        return true;
      };
  auto service = nh.advertiseService("/dra_repro/step", callback);
  ros::spin();
  return 0;
}
