#!/usr/bin/env python3
import math
import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped

# ===== WGS-84 constants =====
_A  = 6378137.0
_F  = 1.0 / 298.257223563
_B  = _A * (1.0 - _F)
_E2 = 1.0 - (_B*_B)/(_A*_A)
_EP2 = (_A*_A)/(_B*_B) - 1.0

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi


# ===== LLA → ECEF =====
def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    s = math.sin(lat)
    c = math.cos(lat)
    N = _A / math.sqrt(1.0 - _E2 * s*s)
    x = (N + h) * c * math.cos(lon)
    y = (N + h) * c * math.sin(lon)
    z = (N * (1 - _E2) + h) * s
    return x, y, z


# ===== ECEF → ENU (relative to lat0, lon0, alt0) =====
def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0)
    dx, dy, dz = x - x0, y - y0, z - z0

    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)
    sl = math.sin(lat0); cl = math.cos(lat0)
    slon = math.sin(lon0); clon = math.cos(lon0)

    t = -slon * dx + clon * dy
    e = t

    t = -sl * clon * dx - sl * slon * dy + cl * dz
    n = t

    t = cl * clon * dx + cl * slon * dy + sl * dz
    u = t

    return e, n, u


class GpsRawToPose:

    def __init__(self):
        rospy.init_node("gps_raw_to_pose", anonymous=True)

        # Set reference origin (lat0, lon0, alt0)
        self.lat0 = rospy.get_param("~lat0", -35.3632583)
        self.lon0 = rospy.get_param("~lon0", 149.1652068)
        self.alt0 = rospy.get_param("~alt0", 0.0)

        rospy.Subscriber("/mavros/global_position/raw",
                         NavSatFix, self.cb, queue_size=1)

        self.pub = rospy.Publisher("/gps_pose",
                                   PoseStamped, queue_size=1)

        rospy.loginfo("gps_raw_to_pose running...")

    def cb(self, gps_msg):
        # Ignore invalid GPS
        if gps_msg.status.status < 0:
            rospy.logwarn_throttle(2.0, "No GPS Fix")
            return

        # Convert LLA -> ECEF
        x, y, z = geodetic_to_ecef(
            gps_msg.latitude,
            gps_msg.longitude,
            gps_msg.altitude
        )

        # Convert ECEF -> ENU relative to reference
        e, n, u = ecef_to_enu(
            x, y, z,
            self.lat0, self.lon0, self.alt0
        )

        # Publish PoseStamped
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"

        ps.pose.position.x = e
        ps.pose.position.y = n
        ps.pose.position.z = u

        # No orientation — GPS provides no yaw
        ps.pose.orientation.w = 1.0

        self.pub.publish(ps)


if __name__ == "__main__":
    GpsRawToPose()
    rospy.spin()
