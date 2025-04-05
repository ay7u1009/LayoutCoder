import cv2
import os
import requests
import json
from base64 import b64encode
import time


def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def ocr_detection_paddle(imgpath):
    from paddleocr import PaddleOCR, draw_ocr
    import cv2

    # 20240823 # modified: 增加det_limit_side_len参数，根据图片自动调整最大长度，防止漏检
    # https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/FAQ.md#q%E5%AF%B9%E4%BA%8E%E4%B8%80%E4%BA%9B%E5%B0%BA%E5%AF%B8%E8%BE%83%E5%A4%A7%E7%9A%84%E6%96%87%E6%A1%A3%E7%B1%BB%E5%9B%BE%E7%89%87%E5%9C%A8%E6%A3%80%E6%B5%8B%E6%97%B6%E4%BC%9A%E6%9C%89%E8%BE%83%E5%A4%9A%E7%9A%84%E6%BC%8F%E6%A3%80%E6%80%8E%E4%B9%88%E9%81%BF%E5%85%8D%E8%BF%99%E7%A7%8D%E6%BC%8F%E6%A3%80%E7%9A%84%E9%97%AE%E9%A2%98%E5%91%A2
    img = cv2.imread(imgpath)
    height, width = img.shape[:2]
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_limit_side_len=max(width, height))
    result = ocr.ocr(imgpath, cls=True)
    return result

def paddle_to_google(paddle_response):
    if not paddle_response or not paddle_response[0]:
        return {"responses": []}

    text_annotations = []
    full_text = ""
    first_bounding_poly = None

    for item in paddle_response[0]:
        bounding_box = item[0]
        text, confidence = item[1]

        # Combine text for the full description
        full_text += text + "\n"

        # Prepare the bounding box in Google OCR format
        bounding_poly = {
            "vertices": [
                {"x": int(vertex[0]), "y": int(vertex[1])} for vertex in bounding_box
            ]
        }

        # Add each text element to textAnnotations
        text_annotations.append({
            "description": text,
            "boundingPoly": bounding_poly
        })

        # Save the first bounding poly for the full text
        if first_bounding_poly is None:
            first_bounding_poly = bounding_poly

    # Remove the trailing newline character from full_text
    if full_text.endswith("\n"):
        full_text = full_text[:-1]

    # Add the combined full text as the first element in textAnnotations
    text_annotations.insert(0, {
        "description": full_text,
        "boundingPoly": first_bounding_poly
    })

    return {
        "responses": [
            {
                "textAnnotations": text_annotations
            }
        ]
    }


def ocr_detection_google(imgpath):
    paddle_result = ocr_detection_paddle(imgpath)
    goole_response = paddle_to_google(paddle_result)
    if len(goole_response['responses'][0]['textAnnotations']) < 2:
        return []
    else:
        return goole_response['responses'][0]['textAnnotations'][1:]
    # # start = time.clock()
    # url = 'https://vision.googleapis.com/v1/images:annotate'
    # api_key = 'AIzaSyDUc4iOUASJQYkVwSomIArTKhE2C6bHK8U'             # *** Replace with your own Key ***
    # imgdata = Google_OCR_makeImageData(imgpath)
    # response = requests.post(url,
    #                          data=imgdata,
    #                          params={'key': api_key},
    #                          headers={'Content_Type': 'application/json'})
    # # print('*** Text Detection Time Taken:%.3fs ***' % (time.clock() - start))
    # print("*** Please replace the Google OCR key at detect_text/ocr.py line 28 with your own (apply in https://cloud.google.com/vision) ***")
    # if 'responses' not in response.json():
    #     raise Exception(response.json())
    # if response.json()['responses'] == [{}]:
    #     # No Text
    #     return None
    # else:
    #     return response.json()['responses'][0]['textAnnotations'][1:]


    # # google.com
    # return [
    #     {'boundingPoly': {'vertices': [{'x': 2048,
    #                                     'y': 38},
    #                                    {'x': 2205,
    #                                     'y': 38},
    #                                    {'x': 2205,
    #                                     'y': 77},
    #                                    {'x': 2048,
    #                                     'y': 77}]},
    #      'description': 'Gmail圖片'},
    #     {'boundingPoly': {'vertices': [{'x': 2248,
    #                                     'y': 44},
    #                                    {'x': 2291,
    #                                     'y': 44},
    #                                    {'x': 2291,
    #                                     'y': 82},
    #                                    {'x': 2248,
    #                                     'y': 82}]},
    #      'description': '!'},
    #     {'boundingPoly': {'vertices': [{'x': 2389,
    #                                     'y': 36},
    #                                    {'x': 2456,
    #                                     'y': 36},
    #                                    {'x': 2456,
    #                                     'y': 80},
    #                                    {'x': 2389,
    #                                     'y': 80}]},
    #      'description': '登入'},
    #     {'boundingPoly': {'vertices': [{'x': 1021,
    #                                     'y': 229},
    #                                    {'x': 1549,
    #                                     'y': 255},
    #                                    {'x': 1542,
    #                                     'y': 401},
    #                                    {'x': 1014,
    #                                     'y': 375}]},
    #      'description': 'Google'},
    #     {'boundingPoly': {'vertices': [{'x': 1109,
    #                                     'y': 633},
    #                                    {'x': 1274,
    #                                     'y': 630},
    #                                    {'x': 1275,
    #                                     'y': 671},
    #                                    {'x': 1110,
    #                                     'y': 674}]},
    #      'description': 'Google 搜寻'},
    #     {'boundingPoly': {'vertices': [{'x': 1355,
    #                                     'y': 632},
    #                                    {'x': 1451,
    #                                     'y': 632},
    #                                    {'x': 1451,
    #                                     'y': 674},
    #                                    {'x': 1355,
    #                                     'y': 674}]},
    #      'description': '好手氣'},
    #     {'boundingPoly': {'vertices': [{'x': 992,
    #                                     'y': 740},
    #                                    {'x': 1341,
    #                                     'y': 740},
    #                                    {'x': 1341,
    #                                     'y': 778},
    #                                    {'x': 992,
    #                                     'y': 778}]},
    #      'description': 'Google 透過以下語言提供：「'},
    #     {'boundingPoly': {'vertices': [{'x': 1325,
    #                                     'y': 737},
    #                                    {'x': 1565,
    #                                     'y': 737},
    #                                    {'x': 1565,
    #                                     'y': 775},
    #                                    {'x': 1325,
    #                                     'y': 775}]},
    #      'description': '中文(简体）English'},
    #     {'boundingPoly': {'vertices': [{'x': 53,
    #                                     'y': 1235},
    #                                    {'x': 125,
    #                                     'y': 1235},
    #                                    {'x': 125,
    #                                     'y': 1279},
    #                                    {'x': 53,
    #                                     'y': 1279}]},
    #      'description': '香港'},
    #     {'boundingPoly': {'vertices': [{'x': 67,
    #                                     'y': 1345},
    #                                    {'x': 131,
    #                                     'y': 1345},
    #                                    {'x': 131,
    #                                     'y': 1378},
    #                                    {'x': 67,
    #                                     'y': 1378}]},
    #      'description': '關於'},
    #     {'boundingPoly': {'vertices': [{'x': 184,
    #                                     'y': 1345},
    #                                    {'x': 245,
    #                                     'y': 1345},
    #                                    {'x': 245,
    #                                     'y': 1378},
    #                                    {'x': 184,
    #                                     'y': 1378}]},
    #      'description': '廣告'},
    #     {'boundingPoly': {'vertices': [{'x': 301,
    #                                     'y': 1345},
    #                                    {'x': 363,
    #                                     'y': 1345},
    #                                    {'x': 363,
    #                                     'y': 1375},
    #                                    {'x': 301,
    #                                     'y': 1375}]},
    #      'description': '企業'},
    #     {'boundingPoly': {'vertices': [{'x': 413,
    #                                     'y': 1342},
    #                                    {'x': 675,
    #                                     'y': 1342},
    #                                    {'x': 675,
    #                                     'y': 1378},
    #                                    {'x': 413,
    #                                     'y': 1378}]},
    #      'description': '搜寻服務的運作方式'},
    #     {'boundingPoly': {'vertices': [{'x': 2115,
    #                                     'y': 1342},
    #                                    {'x': 2264,
    #                                     'y': 1342},
    #                                    {'x': 2264,
    #                                     'y': 1380},
    #                                    {'x': 2115,
    #                                     'y': 1380}]},
    #      'description': '私權政策'},
    #     {'boundingPoly': {'vertices': [{'x': 2317,
    #                                     'y': 1345},
    #                                    {'x': 2379,
    #                                     'y': 1345},
    #                                    {'x': 2379,
    #                                     'y': 1378},
    #                                    {'x': 2317,
    #                                     'y': 1378}]},
    #      'description': '條款'},
    #     {'boundingPoly': {'vertices': [{'x': 2429,
    #                                     'y': 1345},
    #                                    {'x': 2491,
    #                                     'y': 1345},
    #                                    {'x': 2491,
    #                                     'y': 1378},
    #                                    {'x': 2429,
    #                                     'y': 1378}]},
    #      'description': '設定'}]