import cv2
import mediapipe as mp
import numpy as np

img_preta = np.zeros((512, 512), dtype=np.uint8)

def redimenciona_video(largura, altura, caminho_video, caminho_final):
    cap = cv2.VideoCapture(caminho_video)

    new_width = largura
    new_height = altura

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(caminho_final, fourcc, 30, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def captura_labels_video(caminho_video, quantidade_mao, mostra_video=False):
    video = cv2.VideoCapture(caminho_video)

    hand = mp.solutions.hands
    Hand = hand.Hands(max_num_hands=quantidade_mao)
    mpDraw = mp.solutions.drawing_utils

    labels_point = []
    label = {}

    while True:
        check, img = out.read()
        
        if not check:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        handsPoints = results.multi_hand_landmarks
        if handsPoints:
            for points in handsPoints:
                labels_point.append(points)
                # print(points)
                # mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)
                
        if(mostra_video):
            cv2.imshow("Imagem", img)
            cv2.waitKey(1)
    
    #termina video
    out.release()
    cv2.destroyAllWindows()
            
    return labels_point

caminho = "../Videos_original/I love you in sign language. How did I do.mp4"
caminho_saida = "../Video_redimencionado/redMen.mp4"
redimenciona_video(600,450, caminho,caminho_saida)

# l = captura_labels_video(caminho, 1, True)



# print("teste:")

# ls_single_face = l[0].landmark
# cont = 0
# for idx in ls_single_face:
#     print(cont,":","x = ",idx.x, "y = ",idx.y,  "z = ",idx.z)
#     cont+=1
    