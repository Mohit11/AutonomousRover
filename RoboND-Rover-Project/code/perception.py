import numpy as np
import cv2
import matplotlib.image as mpimg
# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
# def color_thresh(img, rgb_thresh=(210, 210, 200)):
def color_thresh(img, rgb_thresh=(160, 160, 160)):
	color_select = np.zeros_like(img[:,:,0])
	above_thresh = (img[:,:,0] > rgb_thresh[0]) \
				& (img[:,:,1] > rgb_thresh[1]) \
				& (img[:,:,2] > rgb_thresh[2])
	color_select[above_thresh] = 1
	return color_select

# def obs(img, rgb_thresh=(200, 50, 50)):
# def obs(img, rgb_thresh=(230, 100, 100)):
def obs(img, rgb_thresh=(230, 100, 100)):
	obs_select = np.zeros_like(img[:,:,0])
	below_thresh = (img[:,:,0] < rgb_thresh[0]) \
				& (img[:,:,1] < rgb_thresh[1]) \
				& (img[:,:,2] < rgb_thresh[2])
	obs_select[below_thresh] = 1
	return obs_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
	# Identify nonzero pixels
	ypos, xpos = binary_img.nonzero()
	# Calculate pixel positions with reference to the rover position being at the 
	# center bottom of the image.  
	x_pixel = 0.7*(-(ypos - binary_img.shape[0]).astype(np.float))
	y_pixel = 0.7*(-(xpos - binary_img.shape[1]/2 ).astype(np.float))
	# good_pixels = np.sqrt(x_pixel**2 + y_pixel**2) < 30
	# x_good = [x for x in good_pixels if x == True]
	# x_pixel = x_pixel[x_good]
	# y_pixel = y_pixel[x_good]
	# x_pixel = filter(lambda item: item.flag, good_pixels)
	# y_pixel = y_pixel[good_pixels]
	return x_pixel, y_pixel

def good_rover_coords(binary_img):
	# Identify nonzero pixels
	ypos, xpos = binary_img.nonzero()
	# Calculate pixel positions with reference to the rover position being at the 
	# center bottom of the image.  
	x_pixel = 0.7*((ypos - binary_img.shape[0]).astype(np.float))
	y_pixel = 0.7*(-(xpos - binary_img.shape[1]/2 ).astype(np.float))
	good_pixels = np.sqrt(x_pixel**2 + y_pixel**2) <= 40
	x_good = [x for x in good_pixels if x == True]
	x_pixel = x_pixel[x_good]
	y_pixel = y_pixel[x_good]
	return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
	# Convert (x_pixel, y_pixel) to (distance, angle) 
	# in polar coordinates in rover space
	# Calculate distance to each pixel
	dist = np.sqrt(x_pixel**2 + y_pixel**2)
	# Calculate angle away from vertical for each pixel
	angles = np.arctan2(y_pixel, x_pixel)
	return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
	# Convert yaw to radians
	yaw_rad = yaw * np.pi / 180
	xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
							
	ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
	# Return the result  
	return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
	# Apply a scaling and a translation
	xpix_translated = (xpix_rot / scale) + xpos
	ypix_translated = (ypix_rot / scale) + ypos
	# Return the result  
	return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
	# Apply rotation
	xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
	# Apply translation
	xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
	# Perform rotation, translation and clipping all at once
	x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
	y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
	# Return the result
	return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
		   
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
	# mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
	# return warped, mask
	return warped

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

	dst_size = 5
	bottom_offset = 6
	img = Rover.img
	source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
	destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
							  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
							  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
							  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
							  ])  

	rock_img = mpimg.imread('../calibration_images/example_rock1.jpg')
	gold = np.uint8([[[140,110,0]]])
	hsv_gold = cv2.cvtColor(gold,cv2.COLOR_BGR2HSV)
	hue = hsv_gold[0][0][0]
	hsv = cv2.cvtColor(rock_img, cv2.COLOR_BGR2HSV)
	lower_gold = np.array([hue-10,100,100])
	upper_gold = np.array([hue+10,255,255])
	gmask = cv2.inRange(hsv, lower_gold, upper_gold)
	res = cv2.bitwise_and(rock_img, rock_img, mask= gmask)

	###################################### With Masking ####################################################

	# warped, mask = perspect_transform(img, source, destination)
	# threshed = color_thresh(img)
	# threshedobs = np.absolute(np.float32(threshed) - 1)*mask
	# Rover.vision_image[:,:,0] = threshedobs*255
	# Rover.vision_image[:,:,1] = gmask*255
	# Rover.vision_image[:,:,2] = threshed*255

	#####################################  Without Masking ##########################################################
	
	# warped = perspect_transform(img, source, destination)
	# threshed = color_thresh(img)
	# threshedobs = np.absolute(np.float32(threshed) - 1)
	# # threshedobs = obs(img)
	# Rover.vision_image[:,:,0] = threshedobs*255
	# Rover.vision_image[:,:,1] = gmask*255
	# Rover.vision_image[:,:,2] = threshed*255

	# xpix, ypix = rover_coords(threshed)
	# xobs, yobs = rover_coords(threshedobs)
	# xgol, ygol = rover_coords(gmask)
	# dist, angles = to_polar_coords(xpix, ypix)


	###################################### Threshing image before warping #######################################################
	
	# threshed = color_thresh(img)
	# threshedobs = np.absolute(np.float32(threshed) - 1)
	# warped = perspect_transform(threshed, source, destination)
	# warpedobs = perspect_transform(threshedobs, source, destination)
	# Rover.vision_image[:,:,0] = warpedobs*255
	# Rover.vision_image[:,:,1] = gmask*255
	# Rover.vision_image[:,:,2] = warped*255
	
	# xpix, ypix = rover_coords(warped)
	# xobs, yobs = rover_coords(warpedobs)
	# xgol, ygol = rover_coords(gmask)
	# dist, angles = to_polar_coords(xpix, ypix)

	# # mean_dir = np.mean(angles)
	# world_size = Rover.worldmap.shape[0]

	# scale = 2*dst_size
					   
	# x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
	# Rover.worldmap[y_world, x_world, 2] += 50

	# # x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
	# # Rover.worldmap[y_world, x_world, 2] = 255

	# xo_world, yo_world = pix_to_world(xobs, yobs, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
	# Rover.worldmap[yo_world, xo_world, 0] += 100
				  
	# xgw, ygw = pix_to_world(xgol, ygol, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
	# Rover.worldmap[ygw, xgw, :] = 255
	
	# Rover.nav_dists = dist			    
	# Rover.nav_angles = angles
					
	# return Rover

	####################################################################################################################################


	# Applying Color Thresholding
	threshed = color_thresh(img)
	# threshedobs = np.absolute(np.float32(threshed) - 1)
	threshedobs = obs(img)
	
	#Applying Perspective Transform
	warped = perspect_transform(threshed, source, destination)
	warpedobs = perspect_transform(threshedobs, source, destination)
	
	# Updating the image with navigable terrain, obstacles and rocks  
	Rover.vision_image[:,:,0] = warpedobs*255
	Rover.vision_image[:,:,1] = gmask*255
	Rover.vision_image[:,:,2] = warped*255
	
	# Calculating map pixel image values for navigable terrain, obstacles and rock samples in Rover Coordinates 
	# xpix, ypix = good_rover_coords(warped)
	xpix, ypix = rover_coords(warped)
	xobs, yobs = rover_coords(warpedobs)
	xgol, ygol = rover_coords(gmask)

	# Converting Rover Centric Pixel positions to Polar Coordinates
	dist, angles = to_polar_coords(xpix, ypix)

	# Converting Rover Centric pixel values to World Coordinates
	world_size = Rover.worldmap.shape[0]
	scale = 2*dst_size
	
	# Navigable pixels
	x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

	# Obstacle pixels
	xo_world, yo_world = pix_to_world(xobs, yobs, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

	# Rock sample pixels
	xgw, ygw = pix_to_world(xgol, ygol, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)


# Updating Rover Worldmap
	# if Rover.pitch < 0.1 :
	# if (Rover.pitch == np.clip(Rover.pitch, -0.1, 0.1) and Rover.roll <= 0.1) :
	Rover.worldmap[y_world, x_world, 2] += 100	
	# else:	
	# Rover.worldmap[y_world, x_world, 2] += 100
	Rover.worldmap[yo_world, xo_world, 0] += 50
	Rover.worldmap[ygw, xgw, :] = 255

	# Updating Rover pixel distances and angles
	Rover.nav_dists = dist			    
	Rover.nav_angles = angles
	# print (" Number of pixels : ", len(Rover.nav_dists))
	return Rover
				
