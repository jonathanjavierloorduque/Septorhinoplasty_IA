import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import mediapipe as mp
import cv2

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
        ####### 1. nose path #######
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            #print(f"ya le arregle {len(self.imgs)}")
            for i in self.imgs:
                print(i)
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing = mp.solutions.drawing_utils
                image = cv2.imread(i)
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5) as face_mesh: 
                    # Process the image
                    results = face_mesh.process(image)
                    height, width,_ = image.shape
                    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    results=face_mesh.process(image_rgb)    
                    #print("Face landmarks:",results.multi_face_landmarks)
                    # Draw the face mesh on the image
                    if results.multi_face_landmarks is not None:    
                        for face_landmarks in results.multi_face_landmarks:
                            #mp_drawing.draw_landmarks(image,face_landmarks)
                            #print(int(face_landmarks.landmark[4].x*width))
                            #print(int(face_landmarks.landmark[4].y*width))
                            x=int(face_landmarks.landmark[221].x*width)
                            y=int(face_landmarks.landmark[221].y*height)
                            h, w = self.image_size
                            print(x,y)
                            mask = bbox2mask(self.image_size, (y, x, h//7, w//9))
            #mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'nose':
            h, w = self.image_size
            #print(f"ya le arregle {len(self.imgs)}")
            array_pointsf=[] 
            for i in self.imgs:
                print(i)
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing = mp.solutions.drawing_utils
                image = cv2.imread(i)
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5) as face_mesh: 
                    # Process the image
                    results = face_mesh.process(image)
                    height, width,_ = image.shape
                    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    results=face_mesh.process(image_rgb)    
                    #print("Face landmarks:",results.multi_face_landmarks)
                    # Draw the face mesh on the image
                    if results.multi_face_landmarks is not None:
                        
                        for face_landmarks in results.multi_face_landmarks:
                            #mp_drawing.draw_landmarks(image,face_landmarks)
                            #print(int(face_landmarks.landmark[4].x*width))
                            #print(int(face_landmarks.landmark[4].y*width))
                            x=int(face_landmarks.landmark[221].x*width)
                            y=int(face_landmarks.landmark[221].y*height)
                            h, w = self.image_size
                            print(x,y)
                            array_pointsf.append((x,y))
            print(array_pointsf)
            for i in range(len(array_pointsf)):
              x=array_pointsf[i][0]
              print("X es=",x)
              y=array_pointsf[i][1]
              print("Y es=",y )
              mask = bbox2mask(self.image_size, (y,x, h//7, w//9))
            #mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'nosegreek':
            h, w = self.image_size
            #print(f"ya le arregle {len(self.imgs)}")
            array_pointsf=[] 
            for i in self.imgs:
                print(i)
                mp_face_detection = mp.solutions.face_detection
                mp_drawing = mp.solutions.drawing_utils
                image = cv2.imread(i)
                with mp_face_detection.FaceDetection(
                    min_detection_confidence= 0.5) as face_detection:
                    height,width,_ = image.shape
                    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    results= face_detection.process(image_rgb)
                
                    if  results.detections is not None:
                        for detection in results.detections:
                          
                          #Ojo Derecho
                          X_RE= int(detection.location_data.relative_keypoints[0].x*width)
                          Y_RE= int(detection.location_data.relative_keypoints[0].y*height)
                        
                          # Ojo izquierdo
                          X_LE= int(detection.location_data.relative_keypoints[1].x*width)
                          Y_LE= int(detection.location_data.relative_keypoints[1].y*height)
                          #nose
                          X_NO= int(detection.location_data.relative_keypoints[2].x*width)
                          Y_NO= int(detection.location_data.relative_keypoints[2].y*height)
                          diff=(X_LE-X_NO)
                          h, w = self.image_size
                          print("La diferencia es entre el ojo izquierzo y la nariz en el eje de las x es",diff)
                          #variable mask width
                          NEW= abs(Y_RE-Y_NO)
                          print("NEW es el ancho de la mascara",NEW)
                          if diff >= 0:
                            if NEW >= 15:  # Agregar una nueva condición basada en NEW
                                # Haz algo si NEW es mayor que 10
                                mask = bbox2mask(self.image_size, (Y_LE + 1, X_NO - 12, NEW + 1, w // 2))
                            else:
                                # Haz algo si NEW es menor o igual que 10
                                mask = bbox2mask(self.image_size, (Y_LE -3, X_NO - 14, h//10, w // 2))
                            
                            #mask = bbox2mask(self.image_size, (Y_LE+1,X_NO-12,NEW+1, w//2))
                            #mask = bbox2mask(self.image_size, (Y_NO-18,X_NO-12, h//10, w//2))
                            #mask = bbox2mask(self.image_size, (Y_NO-15,X_NO-12, h//10, w//2))
                            #mask = bbox2mask(self.image_size, (Y_NO-25,X_NO-13, h//7, w//2))
                            #mask = bbox2mask(self.image_size, (Y_RE+2,23, 255//8+9, 255-50))
                          else:
                            mask = bbox2mask(self.image_size, (Y_LE+1,X_LE+4,NEW+1, w//2))
                            #mask = bbox2mask(self.image_size, (Y_LE,X_LE+4, h//10, w//2))
                            #mask = bbox2mask(self.image_size, (Y_LE+2,X_LE+4, h//10, w//2))
                            #mask = bbox2mask(self.image_size, (Y_LE-5,X_LE+4, h//7, w//2))
                            #mask = bbox2mask(self.image_size, (Y_RE+2,23, 255//8+9, 255-50))            
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


