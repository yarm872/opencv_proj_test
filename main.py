import cv2
import numpy as np

# image=cv2.imread("images\snake.jpeg")
# cv2.imshow("test",image)
# cv2.waitKey(0)


# new_img=np.zeros((300,600,3),dtype="uint8")
# new_img[50:300,50:300]=255,0,0
# cv2.line(new_img,(20,60),(new_img.shape[1]//2,new_img.shape[0]//2),(0,255,0),thickness=1)
# cv2.rectangle(new_img,(50,50),(100,100),(0,0,255),thickness=5)
# cv2.circle(new_img,(new_img.shape[1]//2+100,new_img.shape[0]//2+50),20,(255,255,255),thickness=3)
# cv2.imshow("test",new_img)
# cv2.waitKey(0)


# cap=cv2.VideoCapture("videos\kot.mp4")
# while True:
#     success, img=cap.read()
#     img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
#     #img=cv2.GaussianBlur(img, (21,21),10)
#     img=cv2.Canny(img,60,60)
    
#     kernel=np.ones((3,3),dtype="uint8")
#     img=cv2.dilate(img,kernel,iterations=1)
#     img=cv2.erode(img,kernel,iterations=1)
    
#     cv2.imshow("test",img)
#     if cv2.waitKey(10) & 0xFF == ord("q"):
#         break


# img=cv2.imread("images\snake.jpeg")
# img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# #img=cv2.flip(img,-1)
# def rotate(image, angle):
#     h, w=image.shape[:2]
#     point=(h//2,w//2)
#     matrix=cv2.getRotationMatrix2D(point, angle, 0.5)
#     return cv2.warpAffine(img, matrix, (w,h))

# def transform(image, x, y):
#     h, w=image.shape[:2]
#     matrix=np.float32(
#         [
#             [1,0,x],
#             [0,1,y]
#         ]
#     )
#     return cv2.warpAffine(image,matrix,(w,h))
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=cv2.GaussianBlur(img,(3,3),3)
# img=cv2.Canny(img,8,5)
# con, hir=cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# new_img=np.zeros(img.shape,dtype="uint8")
# new_img=cv2.cvtColor(new_img,cv2.COLOR_GRAY2BGR)
# new_img=cv2.drawContours(new_img,con,-1,(255,0,0),thickness=1)
# cv2.imshow("new",new_img)
# cv2.imshow("img",img)
# cv2.waitKey(0)

# img=cv2.imread("images\snake.jpeg")
# img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# #img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
# r,g,b=cv2.split(img)
# cv2.imshow("test",b)
# cv2.waitKey(0)


# img=np.zeros((300,300),dtype="uint8")
# circle=cv2.circle(img.copy(),(150,150),50,255,-1)
# rectangle=cv2.rectangle(img.copy(),(0,0),(50,50),255,-1)
# img=cv2.bitwise_or(circle,rectangle)
# cv2.imshow("test",img)
# cv2.waitKey(0)
model=cv2.CascadeClassifier("xml/face1.xml")
images=[]
for i in range(1,5):
    images.append(cv2.imread(f'images/im{i}.jpg'))
i=0
for img in images:
    i+=1
    img=cv2.resize(img,(500,500))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    results=model.detectMultiScale(img,1.1,1)

    for (x,y,w,h) in results:
        cv2.rectangle(img,(x,y),(x+w,y+h),0,2)

    cv2.imshow(f"{i}",img)
    cv2.waitKey(0)