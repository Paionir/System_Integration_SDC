#!/usr/bin/env python
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        
        #### MEMEBER VARIABLES SECTION
        self.pose = None
        self.frame_id = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_idx = -1
        self.stopline_dist = 10000
        self.closest_dist = 10000
        self.lights_state = 4  # INITIAL STATE OF TRAFFIC LIGHTS SET TO UNKNOWN
        self.stop_lines_2d = None # STOP LINES POSITIONS FROM PARAM SERVER
        self.stop_lines_tree = None # KDTREE FROM STOP LINES POSITIONS
        self.cruise_mode = 0 # Mode of car cruise (stopped=0, accelerating=1, stopping=2)
        self.last_stop_point = None # Stop position when in stopping state
        self.last_maxspeed_point = None # Position where top speed will be reached when in accelerating mode
        self.max_speed = 11 # max speed TODO: get this from param server
        self.current_speed = None # current velocity get by /current_velocity topic
        # TODO: Add other member variables you need below
        #### END OF MEMBER VARIABLES SECTION


        #### SUBSCRIBERS SECTION:
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        
        # A callback to test reactions of this node to traffic light states
        rospy.Subscriber('vehicle/traffic_lights',TrafficLightArray, self.traffic_test_cb)
         
        # TODO: Add a subscriber for /obstacle_waypoint below
        #### END OF SUBSCRIBERS SECTION


        #### Publishers section
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=0)
        #### End of Publishers section 

        self.parse_stop_lines_params() # GET STOP LINES COORDINATES FROM PARAM SERVER
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.waypoint_tree and self.stop_lines_tree \
            and self.current_speed :

                # get closest waypoint
                self.publish_waypoints()
                rate.sleep()

    # as the function name suggests
    def publish_waypoints(self):
        final_lane = self.generate_lane()
        # for i, wp in enumerate(final_lane.waypoints):
        #     rospy.loginfo('%4d: x: %f vx = %f vz = %f vy = %f',\
        #         i ,wp.pose.pose.position.x, wp.twist.twist.linear.x, wp.twist.twist.angular.z, \
        #         wp.twist.twist.linear.y)
        self.final_waypoints_pub.publish(final_lane)

    # generate the next LOOKAHEAD_WPS waypoints to publish
    def generate_lane(self):

        ####GENERATE NEW PATH####
        lane = Lane()
        lane.header.frame_id = self.frame_id
        lane.header.stamp = rospy.Time.now()

        # get index of waypoint in waypoints tree that is closest to ego car
        pose_x = self.pose.pose.position.x
        pose_y = self.pose.pose.position.y
        self.closest_dist, closest_idx = self.get_closest_waypoint_idx( self.waypoint_tree, self.waypoints_2d,\
                                                     pose_x, pose_y )
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        # get a slice of LOOKAHEAD_WPS base waypoints that are ahead of car
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

             
        ####LOOK FOR DANGER SITUATIONS####
        # get the index of stop line in stop lines tree that is nearest to ego car
        _, closest_stop_idx = self.get_closest_waypoint_idx( self.stop_lines_tree, self.stop_lines_2d,\
                                                          pose_x, pose_y )
        # get position (x,y) of closest stop line 
        stopline_pos = self.stop_lines_2d[closest_stop_idx] 
        # get index of waypoint in waypoints tree closest that is nearest to the stop line position
        _, self.stopline_idx = self.get_closest_waypoint_idx( self.waypoint_tree, self.waypoints_2d,\
                                                                 stopline_pos[0], stopline_pos[1] )

        self.stopline_dist = self.distance(self.base_lane.waypoints, closest_idx, self.stopline_idx)
        wp_dist = self.stopline_idx - closest_idx # distance expressed in waypoint indexes
        
        danger = self.is_danger(self.lights_state)

        if wp_dist > LOOKAHEAD_WPS:
            wp_dist = LOOKAHEAD_WPS

        #### STOPPED MODE LOGIC
        if self.cruise_mode == 0:
            if danger == True:
                # THERE IS DANGER, KEEP STOPPED MODE    
                self.cruise_mode = 0 # keep on stopped state
                regulated_base_waypoints = self.accel_regulate(base_waypoints,wp_dist-4, 0., 0., -1)
                self.last_stop_point = self.stopline_idx
            else:
                # NO DANGER SO WE CAN ACCELERATE TOWARDS TOP SPEED
                self.cruise_mode = 1 # go to accelerating mode
                regulated_base_waypoints = self.accel_regulate(base_waypoints,len(base_waypoints)-1, 0., self.max_speed, 1)
                self.last_maxspeed_point = farthest_idx  

        #### ACCELERATING MODE LOGIC
        elif self.cruise_mode == 1: 
            if danger == True:
                #THERE IS DANGER, GO TO STOPPING MODE
                self.cruise_mode = 2
                regulated_base_waypoints = self.accel_regulate(base_waypoints,wp_dist-4,self.current_speed,0.,-1)
                self.last_stop_point = self.stopline_idx
            else: 
                # NO DANGER SO WE CAN KEEP ACCELERATING TOWARDS TOP SPEED
                self.cruise_mode = 1 # keep on accelerating mode
                #check if last top speed point is too close
                if (self.last_maxspeed_point - closest_idx) > LOOKAHEAD_WPS/4: 
                    final_idx = self.last_maxspeed_point - closest_idx - 1
                else:
                    final_idx = len(base_waypoints) - 1
                    self.last_maxspeed_point = closest_idx + final_idx # set new top speed point
                regulated_base_waypoints = self.accel_regulate(base_waypoints,final_idx, self.current_speed, self.max_speed, 1)

        #### STOPPING MODE LOGIC
        elif self.cruise_mode == 2:
            if danger == True:
                #THERE IS STILL DANGER, KEEP ON STOPPING MODE
                self.cruise_mode = 2
                regulated_base_waypoints = self.accel_regulate(base_waypoints,wp_dist-4,self.current_speed, 0., -1)
                # for i, wp in enumerate(regulated_base_waypoints):
                #     rospy.loginfo('%4d: x: %f vx = %f',\
                #         i ,wp.pose.pose.position.x, wp.twist.twist.linear.x)
                self.last_stop_point = self.stopline_idx
            else:
                #NO MORE DANGER, GO TO ACCELERATING MODE
                self.cruise_mode = 1
                regulated_base_waypoints = self.accel_regulate(base_waypoints,len(base_waypoints)-1, \
                                                                self.current_speed,self.max_speed, 1 )
                self.last_maxspeed_point = farthest_idx

        lane.waypoints = regulated_base_waypoints

        return lane

    def is_danger(self, tl_state):
        
        dist = self.stopline_dist # distance from stop line
        close = max(self.current_speed*self.current_speed/(2.), 15)# distance for smooth safe stop
        tooclose = min(self.current_speed*self.current_speed/(8.),8) # point of no return 

        danger = False
        
        if (dist <= close and dist > tooclose ) and \
            (tl_state == 0 or tl_state == 1 ):
            rospy.loginfo('######### Danger Traffic light at:%8.2f light_state:%2d cruise:%2d ', \
                           self.stopline_dist, self.lights_state, self.cruise_mode)
            danger = True
        
        if (self.cruise_mode == 2 and (tl_state == 0 or tl_state == 1)):
            danger = True
        
        return danger

    def accel_regulate(self, waypoints, final_idx, starting_velocity, final_velocity=0., sign = -1):
        tmp_wps =[]
        
        if sign > 0 and starting_velocity <= 0.01:
            vel_c = starting_velocity + 1.1
        else:
            vel_c = starting_velocity 

        vel_f = final_velocity

        if final_idx > len(waypoints) - 1:
            final_idx =  len(waypoints) - 1
        dist_f = self.distance(waypoints, 0, final_idx) + self.closest_dist
        if dist_f < 1e-5:
            dist_f = 1e5
        
        acc = (vel_f*vel_f - vel_c*vel_c)/(2.*dist_f) # acceleration needed
        if acc > 1.:
            acc = 1.
        elif acc < -5:
            acc = -5

        for i, wp in enumerate(waypoints):
            w = Waypoint()
            w.pose = wp.pose
            w.twist = wp.twist

            dist = self.distance(waypoints, 0, i) + self.closest_dist
            # rospy.loginfo('vel_c: %f dist_f: %f acc: %f',vel_c,dist_f,acc)
            v_dis2= (vel_c*vel_c + 2.*acc*dist)
            if v_dis2 < 0. :
                v_dis2 = 0. 
            v_dis = math.sqrt(v_dis2)
            w.twist.twist.linear.x = v_dis
            
            tmp_wps.append(w)
        
        return tmp_wps

    # this is to smooth the decelarion
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            # Two waypoints back from line so front of car stops at line
            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2* MAX_DECEL * dist)
            if vel < 1.0 :
                vel = 0.0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    # Helper function to find the index of the waypoint closest to (pos_x,pos_y) point
    def get_closest_waypoint_idx(self, tree, points_2d, pos_x, pos_y, ahead = True):
        # get closest x and y coordinates 
        x = pos_x 
        y = pos_y 
        dist, closest_idx = tree.query([x, y],1)

        # check if the closest is ahead or behind the ego vehicle
        if ahead:
            closest_coord = points_2d[closest_idx]
            prev_coord = points_2d[closest_idx - 1]

            # Equation for hyperplane through closest_coords
            cl_vect = np.array(closest_coord)
            prev_vect = np.array(prev_coord)
            pos_vect = np.array([x, y])

            val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
            if val > 0:
                closest_idx = (closest_idx + 1) % len(points_2d)

        return dist, closest_idx

    def pose_cb(self, msg):
        self.pose = msg
        self.frame_id = msg.header.frame_id

    def waypoints_cb(self, waypoints):
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                                for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d) # yay for KDtree

    # Not too sure what to make of this one? 
    def traffic_cb(self, msg):
        rospy.loginfo('TRAFFIC LIGHT STATE: %d',msg.data)
        
    # Callback to test tl_detection reaction using ground truth given by simulator
    def traffic_test_cb(self, msg):
        t_lights = msg.lights
        self.lights_state = t_lights[0].state

    # Helper function to parse /traffic_light_config parameter
    def parse_stop_lines_params(self):
        stop_lines = rospy.get_param('/traffic_light_config')
        index = stop_lines.find('[')
        stop_lines = stop_lines[index:]
        for str in [ '-', ']', '[', ',' ]:
            stop_lines = stop_lines.replace(str,'')
        splitted = stop_lines.split('\n')
        _ = splitted.pop()
        stop_lines_xy = [ [float(coor_array.split()[0]), float(coor_array.split()[1])]  for coor_array in splitted]
        if not self.stop_lines_2d: 
            self.stop_lines_2d = stop_lines_xy
        if not self.stop_lines_tree:
            self.stop_lines_tree = KDTree(self.stop_lines_2d) 

    def velocity_cb(self, msg):
        self.current_speed = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
