import cv2
import numpy as np



# mouse callback function
def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN :
        global count
        print("{} : (x,y) = ({},{})".format(point_place[count],x,y))
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        mask_point.append([y,x])
        count +=1
        print(count)

if __name__ == "__main__":
    global mask_point
    global point_place
    global count
    count=0
    mask_point = []
    point_place = ["left up - left eye","right down - right eye","left up - right eye","right down - right eye","left up - nose", "right down - nose", "left up - mouse", "right down - mouse"]
    file_name="046.jpg"
    # Create a black image, a window and bind the function to window
    img = cv2.imread(file_name)
    img_vis = img.copy()
    mask = np.zeros(img.shape[:2])
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image',mouse_callback, param=None)
    
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27  or count==8:
            masks=[]
            others = np.ones(img.shape[:2])
            for i in range(0,len(mask_point),2):
                parts=mask.copy()
                parts[mask_point[i][0]:mask_point[i+1][0],mask_point[i][1]:mask_point[i+1][1]]=1
                masks.append(parts)
                others*=1-parts
            masks.append(others)
            masks=np.array(masks)
            print(masks.shape)
            np.save("../data/"+file_name.split("/")[-1].split(".")[0]+"_mask.npy",masks)
            for i in range(0,len(mask_point),2):
                cv2.rectangle(img_vis, tuple(mask_point[i][::-1]),tuple(mask_point[i+1][::-1]), (0,0,255), -1)
            cv2.imwrite(file_name.split("/")[-1].split(".")[0]+"_visual.png",img_vis)
            break
    cv2.destroyAllWindows()
