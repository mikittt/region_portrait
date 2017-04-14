import sys
import os
import dlib
import cv2
import numpy as np

def parts_detection(im_path,predictor):

    im = cv2.imread(im_path)
    im_size = im.shape[:2]
    print(im_size)
    mask = np.zeros(im_size)

    dets = detector(im, 1)

    check_dots = False
    check_mask = True
    if len(dets)!=0:
        print("Number of faces detected: {}".format(len(dets)))
        l_eyes=[]
        r_eyes=[]
        noses=[]
        mouses=[]
                
        for k, face in enumerate(dets):
            face = predictor(im, face)
            if check_dots:
                for i in range(0,68):
                    dot=(face.part(i).x,face.part(i).y)
                    cv2.circle(im,dot,1,(0,0,255),-1)
            
            face = np.array([[face.part(i).x, face.part(i).y] for i in range(68)])
            
            ## eye
            eye_x_scale = 4.0
            eye_y_scale = 6.0
            ###  left eye 
            l_eye_center = np.average(face[36:42],axis = 0)
            l_eye = np.vstack([np.array([1-eye_x_scale,1-eye_y_scale])*l_eye_center+np.array([eye_x_scale*face[36][0],eye_y_scale*np.average(face[37:39],axis = 0)[1]]),np.array([1-eye_x_scale,1-eye_y_scale])*l_eye_center+np.array([eye_x_scale*face[39][0],eye_y_scale*np.average(face[40:42],axis = 0)[1]])]).astype(np.uint32)
            l_eye_mask = mask.copy()
            l_eye_mask[l_eye[0][1]:l_eye[1][1],l_eye[0][0]:l_eye[1][0]] = 1
            
            ###  right eye
            r_eye_center = np.average(face[42:48],axis = 0)
            r_eye = np.vstack([np.array([1-eye_x_scale,1-eye_y_scale])*r_eye_center+np.array([eye_x_scale*face[42][0],eye_y_scale*np.average(face[43:45], axis = 0)[1]]),np.array([1-eye_x_scale,1-eye_y_scale])*r_eye_center+np.array([eye_x_scale*face[45][0],eye_y_scale*np.average(face[46:48], axis = 0)[1]])]).astype(np.uint32)
            r_eye_mask = mask.copy()
            r_eye_mask[r_eye[0][1]:r_eye[1][1],r_eye[0][0]:r_eye[1][0]] = 1

            ## nose
            nose_scale=2.0
            nose = np.vstack([(1-nose_scale)*face[29]+nose_scale*np.array([face[31][0],face[27][1]]),(1-nose_scale)*face[29]+nose_scale*np.array([face[35][0],face[33][1]])]).astype(np.uint32)
            nose_mask = mask.copy()
            nose_mask[nose[0][1]:nose[1][1],nose[0][0]:nose[1][0]] = 1
            
            ## mouse
            mouse_x_scale = 1.3
            mouse_y_scale = 2.0
            mouse_center = np.average(face[60:68],axis = 0)
            mouse = np.vstack([np.array([1-mouse_x_scale,1-mouse_y_scale])*mouse_center+np.array([mouse_x_scale*face[48][0],mouse_y_scale*face[50][1]]),np.array([1-mouse_x_scale,1-mouse_y_scale])*mouse_center+np.array([mouse_x_scale*face[54][0],mouse_y_scale*face[57][1]])]).astype(np.uint32)
            mouse_mask = mask.copy()
            mouse_mask[mouse[0][1]:mouse[1][1],mouse[0][0]:mouse[1][0]] = 1

            l_eyes.append(l_eye_mask)
            r_eyes.append(r_eye_mask)
            noses.append(nose_mask)
            mouses.append(mouse_mask)

            if check_mask:
                cv2.rectangle(im,tuple(l_eye[0]),tuple(l_eye[1]),(0,0,255),-1)
                cv2.rectangle(im,tuple(r_eye[0]),tuple(r_eye[1]),(0,0,255),-1)
                cv2.rectangle(im,tuple(nose[0]),tuple(nose[1]),(0,255,0),-1)
                cv2.rectangle(im,tuple(mouse[0]),tuple(mouse[1]),(255,0,0),-1)
            
        l_eyes = np.max(np.array(l_eyes),axis=0)
        r_eyes = np.max(np.array(r_eyes),axis=0)
        noses = np.max(np.array(noses),axis=0)
        mouses = np.max(np.array(mouses),axis=0)

        ## other region
        others = np.ones(im_size)*(1-l_eyes)*(1-r_eyes)*(1-noses)*(1-mouses)
        if check_dots or check_mask:
            cv2.imshow("img",im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        masks = np.vstack((l_eyes,r_eyes,noses,mouses,others)).reshape(-1,im_size[0],im_size[1])
        print(masks.shape)
        np.save(im_path.split("/")[-1].split(".")[0]+"_mask.npy",masks)

if __name__=="__main__":

    predictor_path = os.path.expanduser('~')+"/dlib-19.3/shape_predictor_68_face_landmarks.dat"

    image_path = "sample.jpg"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    parts_detection(image_path,predictor)
