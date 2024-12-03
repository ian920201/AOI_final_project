import cv2
import torch
import torch.nn as nn
from CNN import cnn
from data_process import transfrom_data, get_classes
from PIL import Image

USER = "Ellen Degeneres"

def face_detect(image_path):
    img = cv2.imread(image_path)
    if (img is None):
        print('Error: No image')
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier("./xml/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray)
    return img, faces

def main():
    cuda = torch.cuda.is_available()
    classes = get_classes('./train')
    model = cnn()
    model.load_state_dict(torch.load('./pth/face_CNN.pth'))
    model.eval()
    if (cuda):
        model = model.cuda()
    
    img, faces = face_detect("./test_image/label7.jpg")
    
    target = []
    
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        cv2.imwrite("./process_image/process_image.jpg", face_roi)
        input_image = Image.open("./process_image/process_image.jpg")
        input_tensor = transfrom_data(input_image)
        input_batch = input_tensor.unsqueeze(0)
        if (cuda):
            input_batch = input_batch.cuda()
        with torch.no_grad():
            output = model(input_batch)
            softmax = nn.Softmax(dim=1)
            prob = softmax(output)
            max_prob, predicted = torch.max(prob, 1)
            max_prob = max_prob.item()
            predicted_class = predicted.item()
        if (classes[predicted_class] == USER and max_prob >= 0.8):
            target = [x, y, w, h]
        else:
            face_oil_painting = cv2.xphoto.oilPainting(face_roi, 10, 1, cv2.COLOR_BGR2Lab)
            img[y:y+h, x:x+w] = face_oil_painting
    
    if (len(target) == 0):
        print("No target")
        return  
    
    h, w, _ = img.shape
    w_cut = 300
    h_cut = 250
    x_center = int(target[0] + target[2] / 2)
    y_center = int(target[1] + target[3] / 2)

    x1 = max(0, x_center - w_cut)
    x2 = min(w, x_center + w_cut)
    y1 = max(0, y_center - h_cut)
    y2 = min(h, y_center + h_cut)

    crop_img = img[y1:y2, x1:x2]
    
    cv2.imwrite("./result_image/label7_result.jpg", crop_img)
    cv2.imshow("result", crop_img)
    cv2.waitKey(0)
        
        
        
    
if (__name__ == "__main__"):
    main()