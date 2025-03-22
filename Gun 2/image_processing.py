import cv2
import numpy as np

image_path = "logo.png"
image = cv2.imread(image_path)

if image is None:
    print("Hata: Görüntü dosyası bulunamadı!")
    exit()

cv2.imshow("Orijinal Goruntu ", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

h,w,c = image.shape
print(f"Goruntu Boyutlari: {w}x{h} piksel, {c} kanal (RGB)")

image_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_logo.png", image_grey)
cv2.imshow("Gri Goruntu",image_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_width = 400
new_height = int(image.shape[0] * (new_width / image.shape[1])) 

resized_image = cv2.resize(image, (new_width,new_height))

cv2.imshow("Boyutlandirilmis Goruntu", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

rotate_90 = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("90 Dondurulmus Goruntu", rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped_image = image[50:250,50:250]
cv2.imshow("Kirpilmis Goruntu", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
