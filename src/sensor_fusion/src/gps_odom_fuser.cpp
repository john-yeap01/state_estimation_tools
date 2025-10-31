// Loosely coupled EKF to fuse GNSS with the local position odometry from mavros 
// before feeding into SUPER planner 
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <GeographicLib/LocalCartesian.hpp>

#include <memory>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using Mat9 = Eigen::Matrix<double, 9, 9>;

class KF {
public:
  void initFromOdom(const Eigen::Vector3d& p, const ros::Time& t) {
    if (!inited_) {
      x_.setZero();
      x_.segment<3>(0) = p;  // p0 = odom position; v0=a0=0
      P_.setZero();
      P_.block<3,3>(0,0) = 1e2 * Eigen::Matrix3d::Identity();  // pos var
      P_.block<3,3>(3,3) = 1e3 * Eigen::Matrix3d::Identity();  // vel var
      P_.block<3,3>(6,6) = 1e4 * Eigen::Matrix3d::Identity();  // acc var
    }
    last_t_ = t;
    inited_ = true;
  }

  void predictTo(const ros::Time& t) {
    if (!inited_) return;
    const double dt = (t - last_t_).toSec();
    if (dt > 0.0 && dt < 1.0) predict(dt);
    last_t_ = t;
  }

  void updatePos(const Eigen::Vector3d& z , const Eigen::Matrix3d& R) {
    if (!inited_) return;
    Eigen::Matrix<double, 3, 9> H  = Eigen::Matrix<double, 3, 9>::Zero();
    H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    Eigen::Vector3d y = z - H * x_;  // FIX: innovation uses full x_
    Eigen::Matrix3d S = H * P_ * H.transpose() + R;
    Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * S.inverse();
    x_ = x_ +  K * y;
    Mat9 KH = K * H;
    Mat9 I = Mat9::Identity();
    P_ = (I - KH) * P_ * (I - KH).transpose() + K * R * K.transpose();
  }

  bool inited() { return inited_; }                       // OK as non-const for now
  Eigen::Vector3d getPosition() { return x_.segment<3>(0); }

private:
  void makeFQd(double dt, Mat9& F, Mat9& Qd) {
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
    F.setZero();
    F.block<3,3>(0,0) = I3;
    F.block<3,3>(0,3) = dt * I3;
    F.block<3,3>(0,6) = 0.5 * dt2 * I3;
    F.block<3,3>(3,3) = I3;
    F.block<3,3>(3,6) = dt * I3;
    F.block<3,3>(6,6) = I3;

    Qd.setZero();
    Qd.block<3,3>(0,0) = (dt5/20.0) * I3;
    Qd.block<3,3>(0,3) = (dt4/8.0)  * I3;
    Qd.block<3,3>(0,6) = (dt3/6.0)  * I3;
    Qd.block<3,3>(3,0) = (dt4/8.0)  * I3;
    Qd.block<3,3>(3,3) = (dt3/3.0)  * I3;
    Qd.block<3,3>(3,6) = (dt2/2.0)  * I3;
    Qd.block<3,3>(6,0) = (dt3/6.0)  * I3;
    Qd.block<3,3>(6,3) = (dt2/2.0)  * I3;
    Qd.block<3,3>(6,6) = (dt)       * I3;

    Qd *= q_; // jerk PSD (m^2/s^5)
  }

  void predict(double dt) {
    makeFQd(dt, F_, Qd_);
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Qd_;
  }

  Vec9 x_;
  Mat9 P_;
  double q_ = 0.1;
  ros::Time last_t_;
  bool inited_ = false;
  Mat9 F_;
  Mat9 Qd_;
};

class LIGONode {
public:
  LIGONode (ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh), enu_origin_set_(false) {                      // FIX: init flag here too
    gps_sub_  = nh_.subscribe("/mavros/global_position/global", 10, &LIGONode::gpsCb,  this);
    odom_sub_ = nh_.subscribe("/mavros/local_position/odom",    10, &LIGONode::odom_cb, this);
    kf_ = std::make_shared<KF>();                                    // FIX: actually construct KF
  }

private:
  void gpsCb(const sensor_msgs::NavSatFix::ConstPtr& msg) {
    if (msg->status.status < sensor_msgs::NavSatStatus::STATUS_FIX) return;
    if (!enu_origin_set_) {
      enu_.Reset(msg->latitude, msg->longitude, msg->altitude);
      enu_origin_set_ = true;
      ROS_INFO_STREAM("ENU origin set at lat=" << msg->latitude
                       << " lon=" << msg->longitude
                       << " alt=" << msg->altitude);
    }
    double x, y, z;
    enu_.Forward(msg->latitude, msg->longitude, msg->altitude, x, y, z);
    // enu_.Reverse(x,y,z, msg->latitude, msg->longitude, msg->altitude);e

    if (!kf_ || !kf_->inited()) return;                              // FIX: guard before predict/update in odom cb

    kf_->predictTo(msg->header.stamp);

    Eigen::Vector3d z_enu(x, y, z);

    // measurement noise covariance R
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0,0) = gps_pos_std_xy_ * gps_pos_std_xy_;
    R(1,1) = gps_pos_std_xy_ * gps_pos_std_xy_;
    R(2,2) = gps_pos_std_z_  * gps_pos_std_z_;

    kf_->updatePos(z_enu, R);


    //publish the position -- later can change to odom or tf type
    const auto p = kf_->getPosition();
    ROS_INFO_STREAM_THROTTLE(1.0, "KF pos (ENU): [" << p.transpose() << "]");
  }

  void odom_cb(const nav_msgs::Odometry::ConstPtr& odom_msg) {
    const ros::Time t = odom_msg->header.stamp;
    const Eigen::Vector3d current_pose(odom_msg->pose.pose.position.x,
                                       odom_msg->pose.pose.position.y,
                                       odom_msg->pose.pose.position.z);
    if (kf_ && !kf_->inited()) {
      kf_->initFromOdom(current_pose, t);
    } else if (kf_) {
      kf_->predictTo(t);                                             // FIX: advance on odom too
    }

    ROS_INFO_STREAM("Local position: "
                    << current_pose.x() << ", "
                    << current_pose.y() << ", "
                    << current_pose.z());                            // FIX: log numeric, not geometry type
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber gps_sub_;
  ros::Subscriber odom_sub_;

  std::shared_ptr<KF> kf_;
  GeographicLib::LocalCartesian enu_;
  bool enu_origin_set_;
  double gps_pos_std_xy_{2.0}, gps_pos_std_z_{3.0};
};

int main(int argc, char** argv){
  ros::init(argc,argv,"gps_odom_fuser");
  ros::NodeHandle nh, pnh("~");
  LIGONode node(nh, pnh);
  ros::spin();
  return 0;
}
