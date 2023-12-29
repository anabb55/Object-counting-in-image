
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os
from sklearn.metrics import mean_absolute_error

folder_path = 'pictures2/'
csv_path = 'bulbasaur_count.csv'

photo_names = ['picture_1.jpg',  'picture_2.jpg', 'picture_3.jpg', 'picture_4.jpg', 
               'picture_5.jpg', 'picture_6.jpg', 'picture_7.jpg', 'picture_8.jpg', 'picture_9.jpg', 'picture_10.jpg',]

my_values = []
real_values = []

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  
    for row in csv_reader:
        picture_name, bulbasaur_count = row
        real_values.append(int(bulbasaur_count))

for photo_name, value in zip(photo_names, real_values):
       
    
        img_bulbasaur=cv2.imread(os.path.join(folder_path, photo_name))

        img_bulbasaur_rgb = cv2.cvtColor(img_bulbasaur, cv2.COLOR_BGR2RGB)
                
        lower = np.array([35, 50, 48])
        upper = np.array([90, 255, 250])


        img_bulbasaur_hsv = cv2.cvtColor(img_bulbasaur_rgb, cv2.COLOR_RGB2HSV)

        #metoda je pronadjena u dokumentaciji OpenCV: https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html
        mask = cv2.inRange(img_bulbasaur_hsv, lower, upper)

       
        masked_image = cv2.bitwise_and(img_bulbasaur_rgb, img_bulbasaur_rgb, mask=mask)


        img_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

       
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_filtered = []

        for contour in contours: 
            center, size, angle = cv2.minAreaRect(contour) 
            height, width = size

            if width > 40 and width < 110 and height > 43 and height < 100: 
                contours_filtered.append(contour)

        img_with_contours = img_bulbasaur_rgb.copy()
        cv2.drawContours(img_with_contours, contours_filtered, -1, (255, 0, 0), 2)

        countours_len = len(contours_filtered)
        my_values.append(countours_len)
        print(' %s: %d - %d' % (photo_name, value, countours_len))

        plt.imshow(img_with_contours)
        plt.show()
            



#MAE
mae = mean_absolute_error(real_values, my_values)

print("Ukupan MAE je: {:.2f}".format(mae))




             
        

