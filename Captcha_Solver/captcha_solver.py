from fastai.vision import *
import cv2


def predict(learner, labels, num_slices, folder_path, filename):
    cropped_images_path = cropper(num_slices, folder_path, filename)

    prediction = ''

    for cropped_image_path in cropped_images_path:
        croppedImg = open_image(cropped_image_path)
        _, category, probs = learner.predict(croppedImg)
        prediction += labels.get(category.item())

    return prediction


def cropper(num_slices, folder_path, filename):
    image = cv2.imread(folder_path + filename)

    height, width = image.shape[:2]

    cropped_images_path = []

    for k in range(num_slices):
        delta = int(width / num_slices)
        x1 = k * delta
        x2 = x1 + delta

        croppedImg = image[0:height, x1:x2]
        croppedImg_path = folder_path + filename + str(k) + '.jpg'

        cv2.imwrite(croppedImg_path, croppedImg)
        cropped_images_path.append(croppedImg_path)

    return cropped_images_path


learner = load_learner(os.path.curdir, "export_digit.pkl")
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
          13: 'N', 14: 'O', 15: 'P', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
          23: 'Z'}

prediction = predict(learner, labels, 5, os.path.curdir + '/', "login.png")
print(prediction)
