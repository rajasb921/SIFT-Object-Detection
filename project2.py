import numpy as np
import cv2 as cv

'''
Helper function that uses a nearest neighbours ratio test to iterate through the distances 
b/w descriptors to find the best matching keypoints. Ratio was determined through testing
'''
def ratio_test_match(dist, ratio=0.4):
    matches = []
    for i in range(dist.shape[0]):
        sorted_indices = np.argsort(dist[i])
        nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]
        if dist[i, nearest] < ratio * dist[i, second_nearest]:
            matches.append((i, nearest, dist[i, nearest]))  # Include distance for sorting
    return matches


'''
Helper function to find the angle made by the line passing through points p1, p2
Always returns the angle counterclockwise from the +x axis. Used to determine
orientation of Rocky
'''
def calculate_angle(p1, p2):
    # Calculate the differences in the x and y coordinates
    # Use relative co-ordinates to simplify calculation (p1 is new origin)
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    angle_rad = np.arctan2(abs(y), abs(x))
    angle_deg = np.degrees(angle_rad)     # Convert the angle to degrees
    
    # Check based on quadrant
    if (x >= 0 and y < 0):
        return angle_deg
    elif (x < 0 and y < 0):
        return 180 - angle_deg
    elif (x < 0 and y >= 0):
        return 180 + angle_deg
    else:
        return 360 - angle_deg

'''
Helper function that calculates the Euclidean distance b/w two points p1, p2
Used to determine the scale by which Rocky has been altered
'''
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


'''''''''''''''''''''''''''''''''''''''
    OBJECT DETECTION SCRIPT
'''''''''''''''''''''''''''''''''''''''

filename = input()
ref_img = cv.imread('./reference.png', cv.IMREAD_GRAYSCALE)   
test_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)            

# Resize ref_img to have the same height as test_img
ref_img_height, ref_img_width = ref_img.shape
test_img_height, test_img_width = test_img.shape
scaling_factor = test_img_height / ref_img_height
ref_img = cv.resize(ref_img, (int(ref_img_width * scaling_factor), test_img_height))


# Initiate SIFT detector and compute keypoints, descriptors
sift = cv.SIFT_create(nfeatures = 10000)
kp1, des1 = sift.detectAndCompute(ref_img,None)
kp2, des2 = sift.detectAndCompute(test_img,None)

# Compute distances between descriptors
d1 = des1.reshape((des1.shape[0], 1, des1.shape[1]))
d2 = des2.reshape((1, des2.shape[0], des2.shape[1]))

batch_size = 100    # Go through all descriptors in batches to avoid using excess memory
distances = []
for i in range(0, des1.shape[0], batch_size):
    batch = des1[i:i+batch_size]
    batch = batch.reshape((batch.shape[0], 1, batch.shape[1]))
    dist_batch = np.sum((batch - d2)**2, axis=2)
    distances.append(dist_batch)
dist = np.vstack(distances)

# Ratio Test Match
matches = ratio_test_match(dist)

# Only consider Rocky to be found if there are at least 2 keypoints that are good matches to the reference image
if (len(matches) >= 2):

    # Store keypoints associated with top 10 matches
    top_matches = sorted(matches, key=lambda x: x[2], reverse=True)[:10]
    keypoints1 = []
    keypoints2 = []
    for match in top_matches:
        query_idx, train_idx, _ = match
        keypoints1.append(kp1[query_idx])
        keypoints2.append(kp2[train_idx])


    # Use the 2 best points from keypoints1 & keypoints2
    kp1_first = keypoints1[0].pt
    kp1_second = keypoints1[1].pt

    kp2_first = keypoints2[0].pt
    kp2_second = keypoints2[1].pt

    # Estimate object height (Known height of Rocky was calculated from the resized referenece image)
    scale = euclidean_distance(kp2_first, kp2_second) / euclidean_distance(kp1_first, kp1_second)
    known_height = 1600
    new_height = known_height * scale

    # Orientation = Difference in angles b/w the lines formed by (p1,p2) from ref_img & (p1,p2) from test_img
    angle1 = calculate_angle(kp1_first, kp1_second)
    angle2 = calculate_angle(kp2_first, kp2_second)
    angle_diff = angle2 - angle1
    if (angle_diff < 0):
        angle = -1 * angle_diff
    else:
        angle = 360 - angle_diff

    # Calculate the average x and y coordinates from the top 2 matches to determines the center
    avg_x = (kp2_first[0] + kp2_second[0]) / 2
    avg_y = (kp2_first[1] + kp2_second[1]) / 2
    print(f"{int(round(avg_x))} {int(round(avg_y))} {int(round(new_height))} {int(round(angle))}")

else:
    print("0 0 0 0") # Not found :(