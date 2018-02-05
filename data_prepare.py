import toml
import cv2

cfg_data = toml.load('cfg.toml')

class preprocess_helper():
    def __int__(self):
        self.img_w = cfg_data['preprocess']['image_width']
        self.img_h = cfg_data['preprocess']['image_height']

        self.HE_flag = True if cfg_data['preprocess']['HE_flag'] == 1 else False

        self.interpolation = cfg_data['preprocess']['interpolation']

    def image_enhanced(self, img):
        '''

        :param img: 图片像素矩阵
        :param HE_flag: 是否使用Histogram Equalization
        :return:
        '''

        res = img
        if self.HE_flag:
            res[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            res[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            res[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        return res

    def image_resize(self, img):
        # Image Resizing
        if self.interpolation == 'INTER_CUBIC':
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)

        return img

    def prerocess(self, img):
        img = self.image_enhanced(img)
        img = self.image_resize(img)

        return img


