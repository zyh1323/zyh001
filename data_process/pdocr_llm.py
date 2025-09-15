from paddleocr import PaddleOCR


class PdOCR:
    def __init__(self, det_model_path, rec_model_path):
        self.ocr = PaddleOCR(
                        use_doc_orientation_classify=False,         # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                        use_doc_unwarping=False,                    # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                        use_textline_orientation=False,             # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                        text_detection_model_dir = det_model_path,   # 告诉OCR识别工具，文本检测模型我放在哪儿了
                        text_recognition_model_dir = rec_model_path   # 告诉OCR识别工具，文本识别模型我放在哪儿了
                    )   # 创建ocr识别功能对象
    
    def infer(self, img_path, out_path):
        result = self.ocr.predict(img_path)
        for res in result:                  # 遍历结果
            res.save_to_img(out_path)       # 保存结果图像
            res.save_to_json(out_path)      # 保存结果json文件
        return result


if __name__ == "__main__":
    det_model_path = "../model/PP-OCRv5_server_det"  # 文本检测模型路径
    rec_model_path = "../model/PP-OCRv5_server_rec"  # 文本识别模型路径
    ocr = PdOCR(det_model_path=det_model_path, rec_model_path=rec_model_path)  # 创建一个OCR识别对象
    img_path = "../data/images/002.png"  # 图像路径
    out_path = "../output/ocr_results"  # 结果保存路径
    result = ocr.infer(img_path, out_path)   # 输入一张图像，推理得到结果
    print(result)

