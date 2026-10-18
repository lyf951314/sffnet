import colorsys
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont

from nets.sffnet import SFFNet
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_bbox, postprocess
import cv2


class SFFNetDetector(object):
    _defaults = {

        "model_path"        : os.environ.get("SFFNET_MODEL_PATH", "model_data/sffnet_weights.pth"),
        "classes_path"      : os.environ.get("SFFNET_CLASSES_PATH", "model_data/voc_classes.txt"),

        "backbone"          : 'resnet50',

        "input_shape"       : [512, 512],

        "confidence"        : 0.1,

        "nms_iou"           : 0.3,

        "nms"               : False,

        "letterbox_image"   : False,

        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            


        self.class_names, self.num_classes  = get_classes(self.classes_path)


        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
        show_config(**self._defaults)


    def generate(self, onnx=False):

        assert self.backbone in ['resnet50', 'sffnet']
        if self.backbone in ("resnet50", "sffnet"):
            self.net = SFFNet(num_classes=self.num_classes, pretrained=False)


        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict):
            checkpoint = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in checkpoint.items()
            }
        self.net.load_state_dict(checkpoint, strict=True)
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def detect_image(self, image, image_path, crop=False, count=False):

        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)


            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]


        filename_base = os.path.splitext(os.path.basename(image_path))[0]

        save_dir = os.environ.get("SFFNET_RESULT_DIR", "img_out/sffnet/txt")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        txt_path = os.path.join(save_dir, filename_base + ".txt")


        with open(txt_path, 'w') as f:
            for i in range(len(top_label)):
                class_name = self.class_names[top_label[i]]
                confidence = top_conf[i]
                top, left, bottom, right = top_boxes[i]
                line = f"{image_path} {class_name} {confidence:.2f} {int(left)} {int(top)} {int(right)} {int(bottom)}\n"
                # line = f"{int(0)} {int(left)} {int(top)} {int(right)} {int(bottom)}\n"
                f.write(line)



        for i in range(len(top_label)):
            class_name = self.class_names[top_label[i]]
            confidence = top_conf[i]
            box = top_boxes[i]
            top, left, bottom, right = box



        # font = ImageFont.truetype(font='model_data/simhei.ttf',
        #                           size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        font_path = os.environ.get("SFFNET_FONT_PATH", "model_data/simhei.ttf")
        try:
            font = ImageFont.truetype(font=font_path, size=10)
        except OSError:
            font = ImageFont.load_default()


        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = os.environ.get("SFFNET_CROP_DIR", "img_out/sffnet/crops")
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        draw = ImageDraw.Draw(image)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw.text((left, top - 20), label, fill=self.colors[int(c)], font=font)

            # 手动绘制单像素线条
            for x in range(left, right + 1):
                draw.point((x, top), fill=self.colors[int(c)])
                draw.point((x, bottom), fill=self.colors[int(c)])
            for y in range(top, bottom + 1):
                draw.point((left, y), fill=self.colors[int(c)])
                draw.point((right, y), fill=self.colors[int(c)])

        del draw
        return image

    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
                results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask        = np.zeros((image.size[1], image.size[0]))
        score       = np.max(outputs[0][0].permute(1, 2, 0).cpu().numpy(), -1)
        score       = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score    = (score * 255).astype('uint8')
        mask            = np.maximum(mask, normed_score)
        
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["heatmap", "size", "offset"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)


            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            

            if results[0] is None:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]


        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

