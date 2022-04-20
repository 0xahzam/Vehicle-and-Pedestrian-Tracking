import cv2
#our image
img = "test_car.mp4"
#our pre-trained car classifier
car_classifier = "cars.xml"
pedestrian_classifier = "pedestrian.xml"


#creating our classifier
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

#reading the image
vid = cv2.VideoCapture(img)

#iterate forever over frames
while True:

    #read the current frame
    successful_frame_read, frame = vid.read()

    #converting image into grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #getting function coordinates
    car_coordinates = car_tracker.detectMultiScale(grayscaled_img)
    pedestrian_coordinates = pedestrian_tracker.detectMultiScale(grayscaled_img)
    

    #detecting the car
    for x,y,w,h in car_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
  
    #detecting the pedestrians
    for x,y,w,h in pedestrian_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255),2)


    #display the image
    cv2.imshow("Self Driving",frame)

    #waitkey prevents the window from closing instantaneously, press Q to quit
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

