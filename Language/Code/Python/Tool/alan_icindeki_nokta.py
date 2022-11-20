class calculate_on_plane(object):

    def __init__(self):
        self.INT_MAX = 10000
        
    def onSegment(self, p:tuple, q:tuple, r:tuple) -> bool:
        if ((q[0] <= max(p[0], r[0])) &
            (q[0] >= min(p[0], r[0])) &
            (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    def orientation(self, p:tuple, q:tuple, r:tuple) -> int:
        val = (((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
        if val == 0:
            return 0
        if val > 0:
            return 1
        else:
            return 2 

    def doIntersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        if (o1 != o2) and (o3 != o4):
            return True
        if (o1 == 0) and (self.onSegment(p1, p2, q1)):
            return True
        if (o2 == 0) and (self.onSegment(p1, q2, q1)):
            return True
        if (o3 == 0) and (self.onSegment(p2, p1, q2)):
            return True
        if (o4 == 0) and (self.onSegment(p2, q1, q2)):
            return True
        return False
    
    def is_inside_polygon(self, points:list, p:tuple) -> bool:
        n = len(points)
        if n < 3:
            return False

        extreme = (self.INT_MAX, p[1])
        count = i = 0
        
        while True:
            next = (i + 1) % n
            
            if (self.doIntersect(points[i],points[next], p, extreme)):                 
                if self.orientation(points[i], p, points[next]) == 0:
                    return self.onSegment(points[i], p, points[next])                 
                count += 1
            i = next
        
            if (i == 0):
                break
            
        return (count % 2 == 1)


if __name__ == '__main__':
    cal = calculate_on_plane()
    
    polygon = [ (50, 50), (60, 100), (100, 90), (80, 60) ]
    p = (0, 0)
    
    print(cal.is_inside_polygon(points = polygon, p = p))
