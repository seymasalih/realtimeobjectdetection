import cv2
import numpy as np





webcam_video_stream = cv2.VideoCapture(0) #video yakalama nesneni


while True:
    
    
    #verilen video dosyasını resim halinde alıp while sokuyoruz
    ret,current_frame = webcam_video_stream.read()
    img = current_frame


    classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
            "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
            "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    


    colors = np.random.uniform(0, 255, size=(len(classes), 3)) #classes verisini çizdirirken yardımcı bi fonksiyon

    net = cv2.dnn.readNetFromDarknet("Models/yolov3.cfg", "Models/yolov3.weights") #yolo'nun önceden eğitilmiş modelin
    #ağırlık dosyasını ve onu yapılandırmak için olan configürasyon dosyasını ekleyelim.

    layer_names = net.getLayerNames() #ağdaki tümkatmanların adını alır

    output_layer = [layer_names[199], layer_names[226], layer_names[253]]#çıktı katmanını alırız

    height, width, channels = img.shape
    #yolonun dosyalarına ekliyoruz
    #görüntüden 4 boyutlu blob oluşturur.
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer) 



    #gerekli olan 3 öğeyi ekliyoruz bunlar sınıf, güvenilirlik , box
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 2)


    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

