import numpy as np
import cv2

def nothing(x):
    pass


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    if (not cap.isOpened()):
        CV_Assert("Cam open failed");

    cap.set(3,1280);
    cap.set(4,1024);

    coeff = 1.5
    
    cv2.namedWindow('TaskBar')

    # create trackbars for color change
    cv2.createTrackbar('B','TaskBar',0, 255, nothing)
    cv2.createTrackbar('G','TaskBar',0, 255, nothing)
    cv2.createTrackbar('R','TaskBar',0, 255, nothing)
    
    kernel = np.ones((5,5), np.uint8) 

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        bgrImage = cv2.flip(frame, -1)
        
        edges = cv2.Canny(bgrImage,100,200)
        edges = cv2.dilate(edges, kernel, iterations=1)
        Nedges = 255 - edges
        
        # hsvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)
           
        # bgrImage[:, :, 0] =  cv2.add(cv2.getTrackbarPos('B','TaskBar') * cv2.bitwise_and(bgrImage[:, :, 0], edges), cv2.bitwise_and(bgrImage[:, :, 0], Nedges)) 
        # bgrImage[:, :, 1] = cv2.add(cv2.getTrackbarPos('G','TaskBar') * cv2.bitwise_and(bgrImage[:, :, 1], edges), cv2.bitwise_and(bgrImage[:, :, 1], Nedges))
        # bgrImage[:, :, 2] = cv2.add(cv2.getTrackbarPos('R','TaskBar') * cv2.bitwise_and(bgrImage[:, :, 2], edges), cv2.bitwise_and(bgrImage[:, :, 2], Nedges))
        
        print("before")
        print(bgrImage[512, 15,  0])
        
        cv2.multiply(bgrImage[:, :, 0], cv2.getTrackbarPos('B','TaskBar'), bgrImage[:, :, 0], 1, cv2.CV_32S);
        # bgrImage[:, :, 0] = cv2.getTrackbarPos('B','TaskBar') * bgrImage[:, :, 0]
        # bgrImage[:, :, 1] = cv2.getTrackbarPos('G','TaskBar') * bgrImage[:, :, 1]
        # bgrImage[:, :, 2] = cv2.getTrackbarPos('R','TaskBar') * bgrImage[:, :, 2]
        
        print("after")
        print(bgrImage[512, 15,  0])
                
        # bgrImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)      
      
        # Display the resulting frame
        cv2.imshow('frame', bgrImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.imwrite( "bgr.png", bgrImage);
    cv2.imwrite( "edge.png", edges);
    cv2.imwrite( "nedge.png", Nedges);

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows() 