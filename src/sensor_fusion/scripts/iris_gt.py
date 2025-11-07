#!/usr/bin/env python3
# Takes in gazebo model state pose of the iris 
# to publish pose and gps of the gazebo ground truth
import math
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus

# ==== WGS-84 constants ====
_A  = 6378137.0                      # semi-major axis [m]
_F  = 1.0 / 298.257223563            # flattening
_B  = _A * (1.0 - _F)                # semi-minor axis [m]
_E2 = 1.0 - (_B*_B)/(_A*_A)          # first eccentricity^2
_EP2 = (_A*_A)/(_B*_B) - 1.0         # second eccentricity^2

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def geodetic_to_ecef(lat_deg, lon_deg, h):
    """LLA -> ECEF (meters), WGS-84."""
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    s = math.sin(lat); c = math.cos(lat)
    N = _A / math.sqrt(1.0 - _E2 * s*s)
    x = (N + h) * c * math.cos(lon)
    y = (N + h) * c * math.sin(lon)
    z = (N * (1.0 - _E2) + h) * s
    return x, y, z

def enu_to_ecef(e, n, u, lat0_deg, lon0_deg, h0):
    """ENU at origin -> ECEF, WGS-84."""
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0)
    lat0 = deg2rad(lat0_deg); lon0 = deg2rad(lon0_deg)
    sl = math.sin(lat0); cl = math.cos(lat0)
    slon = math.sin(lon0); clon = math.cos(lon0)
    # ENU->ECEF rotation at origin
    R = [
        [-slon,            clon,           0.0],
        [-sl*clon,        -sl*slon,        cl ],
        [ cl*clon,         cl*slon,        sl ]
    ]
    dx = R[0][0]*e + R[0][1]*n + R[0][2]*u
    dy = R[1][0]*e + R[1][1]*n + R[1][2]*u
    dz = R[2][0]*e + R[2][1]*n + R[2][2]*u
    return x0 + dx, y0 + dy, z0 + dz

def ecef_to_geodetic(x, y, z):
    """ECEF -> LLA (degrees, degrees, meters), WGS-84. Bowring-like closed form."""
    p = math.hypot(x, y)
    if p == 0.0:
        lon = 0.0
        lat = math.copysign(math.pi/2.0, z)
        h = abs(z) - _B
        return rad2deg(lat), rad2deg(lon), h

    theta = math.atan2(z * _A, p * _B)
    st = math.sin(theta); ct = math.cos(theta)
    lat = math.atan2(z + _EP2 * _B * st**3, p - _E2 * _A * ct**3)
    lon = math.atan2(y, x)
    s = math.sin(lat); N = _A / math.sqrt(1.0 - _E2 * s*s)
    h = p / math.cos(lat) - N
    return rad2deg(lat), rad2deg(lon), h

class GT_SUB:
    def __init__(self):
        rospy.init_node('iris_ground_truth_pose', anonymous=True)

        # Params
        self.target  = rospy.get_param("~model_name", "final_iris")
        self.lat0    = rospy.get_param("~lat0", -35.3632583)
        self.lon0    = rospy.get_param("~lon0", 149.1652068)
        self.alt0    = rospy.get_param("~alt0", 0.0)
        self.frame   = rospy.get_param("~frame_id", "world")
        self.yaw_off = rospy.get_param("~enu_yaw_deg", 0.0)  # rotate ENU by this yaw before conversion

        # Covariance (set small nonzero if you want 'noisy' GPS)
        cov_diag = rospy.get_param("~cov_diag", [0.0, 0.0, 0.0])
        self.cov = [
            float(cov_diag[0]), 0.0, 0.0,
            0.0, float(cov_diag[1]), 0.0,
            0.0, 0.0, float(cov_diag[2]),
        ]
        self.cov_type = rospy.get_param("~cov_type", NavSatFix.COVARIANCE_TYPE_KNOWN)

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb, queue_size=1)
        self.pose_pub = rospy.Publisher("/iris_gt_pose", PoseStamped, queue_size=1)
        self.gps_pub  = rospy.Publisher("/iris_gt_gps",  NavSatFix,   queue_size=1)

    def cb(self, msg: ModelStates):
        try:
            i = msg.name.index(self.target)
        except ValueError:
            rospy.logwarn_throttle(2.0, "%s not in /gazebo/model_states", self.target)
            return

        # Gazebo pose (assumed ENU-aligned world)
        pos = msg.pose[i].position
        e, n, u = pos.x, pos.y, pos.z

        # Optional yaw offset between Gazebo world X/Y and true East/North
        if self.yaw_off:
            c = math.cos(deg2rad(self.yaw_off)); s = math.sin(deg2rad(self.yaw_off))
            e, n = c*e - s*n, s*e + c*n

        # ENU -> ECEF -> LLA
        xe, ye, ze = enu_to_ecef(e, n, u, self.lat0, self.lon0, self.alt0)
        lat, lon, alt = ecef_to_geodetic(xe, ye, ze)

        now = rospy.Time.now()

        # Publish PoseStamped for RViz sanity
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = self.frame
        ps.pose = msg.pose[i]
        self.pose_pub.publish(ps)

        # Publish NavSatFix
        fix = NavSatFix()
        fix.header.stamp = now
        fix.header.frame_id = self.frame
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = (NavSatStatus.SERVICE_GPS |
                              NavSatStatus.SERVICE_GLONASS |
                              NavSatStatus.SERVICE_GALILEO)

        fix.latitude  = float(lat)
        fix.longitude = float(lon)
        fix.altitude  = float(alt)

        fix.position_covariance = self.cov
        fix.position_covariance_type = self.cov_type

        self.gps_pub.publish(fix)

if __name__ == "__main__":
    GT_SUB()
    rospy.spin()
