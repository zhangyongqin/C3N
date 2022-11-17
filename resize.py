from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

house = Image.open('9.jpg')
house_r = house.resize((256,256))

house_r.save('9_256.jpg') # 保存修改后的图片
