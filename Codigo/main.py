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


def criar_video_apenas_do_esqueleto_da_mao(caminho_video, novo_nomeArquivo,quantidade_mao, mostra_video=False, mostra_imagem_presta=True):
    video = cv2.VideoCapture(caminho_video)

    hand = mp.solutions.hands
    Hand = hand.Hands(max_num_hands=quantidade_mao)
    mpDraw = mp.solutions.drawing_utils
    
    
    new_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    caminho_saida = "../Video_redimencionado/"+novo_nomeArquivo+".mp4"

    # Pre-processamento para criar a imagem
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(caminho_saida, fourcc, 30, (new_width, new_height))

    while True:
        check, img = video.read()

        if not check:
            break
        # imagem preta de fundo onde sera desenhada a nova imagem
        if mostra_imagem_presta:
            img_preta = np.zeros(img.shape, dtype=np.uint8)
        else:
            img_preta = img

        # Processo onde é desenhado pego os pontos em relação a mao e desenhado na imagem preta
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        handsPoints = results.multi_hand_landmarks  #Pegar os pontos
        if handsPoints:
            for points in handsPoints:
                # print(points)
                mpDraw.draw_landmarks(img_preta, points, hand.HAND_CONNECTIONS)
        
        #Mostra video
        if (mostra_video):
            cv2.imshow("Video", img_preta)
            cv2.waitKey(1)
        
        # Salvando frame por frame
        resized_frame = cv2.resize(img_preta, (new_width, new_height))
        out.write(resized_frame)

    # termina video
    video.release()
    out.release()
    cv2.destroyAllWindows()


caminho = "../Video_original/teste10.mp4"
# redimenciona_video(1280 , 720, caminho, caminho_saida)

criar_video_apenas_do_esqueleto_da_mao(caminho, "teste10",2, True,mostra_imagem_presta=False)
