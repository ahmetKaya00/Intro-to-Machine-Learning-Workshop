import cv2
image = cv2.imread("logo.png")

blue_channel,green_channel,red_channel = cv2.split(image)

cv2.imshow("Mavi Kanal",blue_channel)
cv2.imshow("Yeşil Kanal",green_channel)
cv2.imshow("Kırmızı Kanal",red_channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

modified_image = cv2.merge([green_channel,red_channel,blue_channel])
cv2.imshow("Degistirilmis Kanal",modified_image)

cv2.waitKey(0)
cv2.destroyAllWindows()