import cv2
import numpy as np



# mouse callback function
def mouse_callback(event,x,y,flags,param):
    global count
    global face_num
    if event == cv2.EVENT_LBUTTONDOWN :
        print("{} : (x,y) = ({},{})".format(point_place[count],x,y))
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        mask_point.append([y,x])
        face_num += 1
    if event == cv2.EVENT_RBUTTONDOWN :
        count += 1
        print(count)

if __name__ == "__main__":
    global mask_point
    global point_place
    global count
    global face_num
    face_num=0
    count=0
    mask_point = []
    point_place = ["left eye","right eye", "nose", "mouse","face"]
    file_name="data/11.jpg"
    # Create a black image, a window and bind the function to window
    img = cv2.imread(file_name)
    img_vis = img.copy()
    mask = np.zeros(img.shape[:2])
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image',mouse_callback, param=None)
    
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27  or count==5:
            part_num = face_num//len(point_place)
            print(part_num)
            masks    = []
            gather   = []
            others   = np.ones(img.shape[:2])
            for i in range(0,len(mask_point),2):
                print(i)
                parts = mask.copy()
                parts[mask_point[i][0]:mask_point[i+1][0],mask_point[i][1]:mask_point[i+1][1]]=1
                gather.append(parts)
                if (i+2) % part_num == 0 :
                    print("ok")
                    gather = np.max(np.array(gather),axis=0)
                    if i<=len(mask_point)-part_num-2:
                        masks.append(gather)
                        others *= 1-gather
                        gather  = []
                    else:
                        masks.append(gather*others)
                        others *= 1-gather
                        masks.append(others)
            masks=np.array(masks)
            print(masks.shape)
            np.save("data/"+file_name.split("/")[-1].split(".")[0]+"_mask.npy",masks)
            cv2.imwrite(file_name.split("/")[-1].split(".")[0]+"_other_face.png",masks[4].astype(np.uint8)*255)
            cv2.imwrite(file_name.split("/")[-1].split(".")[0]+"_other.png",masks[5].astype(np.uint8)*255)
            for i in range(0,len(mask_point),2):
                cv2.rectangle(img_vis, tuple(mask_point[i][::-1]),tuple(mask_point[i+1][::-1]), (0,0,255), 1)
            cv2.imwrite(file_name.split("/")[-1].split(".")[0]+"_visual.png",img_vis)
            break
    cv2.destroyAllWindows()
