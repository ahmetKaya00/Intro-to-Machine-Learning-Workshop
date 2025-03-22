import cv2
import pygame
from deepface import DeepFace

cap = cv2.VideoCapture(0)

pygame.mixer.init()

duygu_cevirisi = {
    "angry": "Kizgin",
    "disgust": "Tiksinti",
    "fear": "Korku",
    "happy": "Mutlu",
    "sad": "Uzgun",
    "surprise": "Saskin",
    "neutral": "Notr"
}
duygu_sesleri = {
    "happy": "happy.mp3",
    "sad": "sad.mp3"
}

son_duygu = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        analiysis = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)

        for face in analiysis:
            x,y,w,h = face['region']['x'],face['region']['y'],face['region']['w'],face['region']['h']

            emotion = face['dominant_emotion']
            turkce_emotion = duygu_cevirisi.get(emotion,emotion)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,turkce_emotion,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(30,255,12),2)

            if emotion in duygu_sesleri and emotion != son_duygu:
                pygame.mixer.music.load(duygu_sesleri[emotion])
                pygame.mixer.music.play()
                son_duygu = emotion
    except Exception as e:
        print("Hata",e)

    cv2.imshow("Gercek Zamanli Yuz Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()