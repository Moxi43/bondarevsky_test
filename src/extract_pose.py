def pose_to_points(img, pose):
    '''Extract coordinates of landmark points in images'''
    points = []
    if pose.pose_landmarks == None:
        return points
    for id, lm in enumerate(pose.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        points.append((cx, cy))
    return points