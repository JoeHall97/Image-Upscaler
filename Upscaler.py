import cv2
from cv2 import dnn_superres
import sys

models = ["edsr", "espcn", "fsrcnn", "lapsrn"]

def upscale(image_path, model_name, model_path, scale, output_path='./upscaled.png'):
  if model_name not in models:
    print('{0} not a valid model'.format(model_name))
    return
  sr = dnn_superres.DnnSuperResImpl_create()
  
  image = cv2.imread(image_path)
  sr.readModel(model_path)

  # set the desired model and scale
  sr.setModel(model_name,scale)

  upscaled_image = sr.upsample(image)

  cv2.imwrite(output_path, upscaled_image)

if __name__ == "__main__":
  # valid models: EDSR, ESPCN, FSRCNN, LapSRN
  if len(sys.argv)!=5:
    print("Usage: python3 Upscaler.py <image_path> <model_path> <model_name> <model_scaler>")
    sys.exit(1)
  upscale(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4])