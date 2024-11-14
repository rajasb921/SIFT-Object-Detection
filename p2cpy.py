import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def ratio_test_match(dist, ratio=0.25):
    matches = []
    for i in range(dist.shape[0]):
        sorted_indices = np.argsort(dist[i])
        nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]
        if dist[i, nearest] < ratio * dist[i, second_nearest]:
            matches.append((i, nearest, dist[i, nearest]))  # Include distance for sorting
    return matches

def calculate_angle(p1, p2):
    # Calculate the differences in the x and y coordinates
    # Use relative co-ordinates to simplify calculation (p1 is 0,0)
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    # Calculate the angle in radians using arctangent of dy/dx
    angle_rad = np.arctan2(abs(y), abs(x))
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    # Check based on quadrant
    if (x >= 0 and y < 0):
        return angle_deg
    elif (x < 0 and y < 0):
        return 180 - angle_deg
    elif (x < 0 and y >= 0):
        return 180 + angle_deg
    else:
        return 360 - angle_deg

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

filename = input()
img1 = cv.imread('./reference.png', cv.IMREAD_GRAYSCALE)   # queryImage
img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)            # trainImage

# Resize img1 to have the same height as img2
img1_height, img1_width = img1.shape
img2_height, img2_width = img2.shape

# Calculate the scaling factor
scaling_factor = img2_height / img1_height

# Resize img1
img1 = cv.resize(img1, (int(img1_width * scaling_factor), img2_height))

# Initiate SIFT detector
sift = cv.SIFT_create(nfeatures = 5000)
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

d1 = des1.reshape((des1.shape[0], 1, des1.shape[1]))
d2 = des2.reshape((1, des2.shape[0], des2.shape[1]))

batch_size = 100
distances = []
for i in range(0, des1.shape[0], batch_size):
    batch = des1[i:i+batch_size]
    batch = batch.reshape((batch.shape[0], 1, batch.shape[1]))
    dist_batch = np.sum((batch - d2)**2, axis=2)
    distances.append(dist_batch)
dist = np.vstack(distances)

# Ratio Test Match
matches = ratio_test_match(dist)

if (len(matches) >= 2):
    # Sort matches by distance in reverse order and take top 10
    top_matches = sorted(matches, key=lambda x: x[2], reverse=True)[:10]

    # Store keypoints associated with top 10 matches
    keypoints1 = []
    keypoints2 = []

    for match in top_matches:
        query_idx, train_idx, _ = match
        keypoints1.append(kp1[query_idx])
        keypoints2.append(kp2[train_idx])

    # # Draw matches
    # img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, 
    #                              [cv.DMatch(i, i, 0) for i in range(len(keypoints1))], None, 
    #                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Display the matches
    # plt.imshow(img_matches)
    # plt.title('Top 10 Keypoint Matches')

    # # Draw circles around the best matches
    # if len(top_matches) >= 2:
    #     # Adjust keypoints2 coordinates (since img1 and img2 are concatenated horizontally)
    #     img1_width = img1.shape[1]
        
    #     # First keypoint in img2
    #     img_matches = cv.circle(img_matches, 
    #                 (int(keypoints2[0].pt[0] + img1_width), int(keypoints2[0].pt[1])), 
    #                 10, (0, 255, 0), 2)
    #     # Second keypoint in img2
    #     img_matches = cv.circle(img_matches, 
    #                 (int(keypoints2[1].pt[0] + img1_width), int(keypoints2[1].pt[1])), 
    #                 10, (0, 255, 0), 2)
    #     # First keypoint in img1
    #     img_matches = cv.circle(img_matches, 
    #                 (int(keypoints1[0].pt[0]), int(keypoints1[0].pt[1])), 
    #                 10, (255, 0, 0), 2)
    #     # Second keypoint in img1
    #     img_matches = cv.circle(img_matches, 
    #                 (int(keypoints1[1].pt[0]), int(keypoints1[1].pt[1])), 
    #                 10, (255, 0, 0), 2)

    # plt.imshow(img_matches)
    # plt.title('Top 10 Keypoint Matches')
    # plt.show()

    # Use the 2 best points from keypoints1 & keypoints2
    kp1_first = keypoints1[0].pt
    kp1_second = keypoints1[1].pt

    kp2_first = keypoints2[0].pt
    kp2_second = keypoints2[1].pt

    # Estimate object height
    scale = euclidean_distance(kp2_first, kp2_second) / euclidean_distance(kp1_first, kp1_second)
    known_height = 1400
    new_height = known_height * scale

    # Use only the top 2 keypoints for angle calculation
    angle1 = calculate_angle(kp1_first, kp1_second)
    angle2 = calculate_angle(kp2_first, kp2_second)

    # Calculate the difference in angles
    angle_diff = angle2 - angle1
    if (angle_diff < 0):
        angle = -1 * angle_diff
    else:
        angle = 360 - angle_diff

    # Calculate the average x and y coordinates from the top 10 matches
    avg_x = np.mean([kp.pt[0] for kp in keypoints2])
    avg_y = np.mean([kp.pt[1] for kp in keypoints2])

    # Print out the average x, y, height, and angle
    print(f"{int(round(avg_x))} {int(round(avg_y))} {int(round(new_height))} {int(round(angle))}")
else:
    # # Store keypoints associated with top 10 matches
    # keypoints1 = []
    # keypoints2 = []
    # # Draw matches
    # img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, 
    #                              [cv.DMatch(i, i, 0) for i in range(len(keypoints1))], None, 
    #                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Display the matches
    # plt.imshow(img_matches)
    # plt.title('Top 10 Keypoint Matches')

    # plt.show()
    print("0 0 0 0")