import cv2
import mediapipe as mp
import numpy as np

def redimenciona_video(largura, altura, caminho_video, caminho_final):
    cap = cv2.VideoCapture(caminho_video)

    new_width = largura
    new_height = altura

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
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

    while True:
        check, img = video.read()

        img_preta = np.zeros(img.shape, dtype=np.uint8)

        if not check:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        handsPoints = results.multi_hand_landmarks
        if handsPoints:
            for points in handsPoints:
                # print(points)
                mpDraw.draw_landmarks(img_preta, points, hand.HAND_CONNECTIONS)

        if (mostra_video):
            cv2.imshow("Video", img_preta)
            cv2.waitKey(1)

    # termina video
    video.release()
    cv2.destroyAllWindows()


caminho = "../Video_original/teste.mp4"
caminho_saida = "../Video_redimencionado/redMen.mp4"
# redimenciona_video(600, 450, caminho, caminho_saida)

captura_labels_video(caminho, 1, True)
