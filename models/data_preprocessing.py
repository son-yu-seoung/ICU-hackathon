import argparse
import os
import cv2
import numpy as np

from scripts.estimator import TfPoseEstimator
from scripts.networks import get_graph_path, model_wh
import scripts.action_classification as act_class

base_dir = './'

if __name__ == '__main__':
    # 각종 argument들을 받음.
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='datasets/A.mp4')
    parser.add_argument('--xml-label', type=str, default='datasets/A.xml')
    parser.add_argument('--category', type=str, default='train')

    parser.add_argument('--resize', type=str, default='0x0')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0)

    parser.add_argument('--model', type=str, default='mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False)
    args = parser.parse_args()
    
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    video_dir = base_dir + args.video
    video = cv2.VideoCapture(video_dir)
    ret_val, frame = video.read()

    video_width_limit = 4000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    action_graph = act_class.graph

    video_name = video_dir.replace('/', '|').replace('\\', '|').replace('.', '|').split('|')[-2]

    output_img_dir = os.path.join(base_dir, 'datasets', args.category, 'preproceed_image', video_name)

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    frame_number = 0
    while video.isOpened():
        if frame_number % 100 == 0:
            print(frame_number)
        ret_val, frame = video.read()
        frame_number += 1
        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        # TfPoseEstimator에서 PreTrain 된 model을 통해 사람의 skeleton point를 찾아냄
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # skeleton point를 통해 한 프레임에서 사람들의 범위를 상하좌우 픽셀단위로 변환함.
        boundarys = TfPoseEstimator.get_humans_imgbox(frame, humans, 0.02)

        # boundarys : {사람1 : [left, right, up, down], 사람2 : [left, right, up, down], ... }
        for num_of_human in boundarys:
            # 각 변수들을 더 직관적으로 임시저장
            img_left = boundarys[num_of_human][0]
            img_right = boundarys[num_of_human][1]
            img_up = boundarys[num_of_human][2]
            img_down = boundarys[num_of_human][3]
            
            # 프레임 크기와 동일한 흰 바탕의 이미지를 그린다.
            ske_img = np.zeros(frame.shape,dtype=np.uint8)
            ske_img.fill(255) 

            # 해당 이미지 위에다가 num_of_human 번째 사람의 skeleton 이미지를 그린다.
            ske_img = TfPoseEstimator.draw_humans(ske_img, [humans[num_of_human]], imgcopy=True)

            # 흰 바탕의 스켈레톤 이미지 그려진 부분만 따서 프레임넘버_사람 넘버.jpg로 스켈레톤 이미지 폴더에 저장한다.
            cv2.imwrite(os.path.join(output_img_dir, "%s_%05d_%02d.jpg" % (video_name, frame_number, num_of_human)), ske_img[img_up:img_down, img_left:img_right])
    
    video.release()

    cv2.destroyAllWindows()