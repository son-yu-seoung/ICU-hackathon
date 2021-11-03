'''
predict.py : 동영상 파일에서 사람의 스켈레톤 이미지를 추출하여 동작을 판단하는 python 파일
대표적인 argument
--video : 동영상 파일 (확장자까지)
'''
import argparse
import logging
import time
import os

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.action_classification as act_class

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    # 각종 argument들을 받음.
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='testvideo.mp4')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--labels', type=str, default='models/retrained_mobilenet_models/trained_model/output_labels.txt')
    args = parser.parse_args()




    # 이미지에서 스켈레톤 이미지를 뽑아주는 OpenPose모델을 불러오는  부분
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))




    # 모델의 라벨들을 불러옴
    labels = act_class.load_labels(args.labels)

    # 불러온 라벨들로 이상행동을 탐지하기 위한 상태변수 추가
    human_action_detection = {}
    for label in labels[1:]: # 0001(정상) 말고 나머지에 대해서
        # 시작 카운트, 시작 프레임넘버, 종료 카운트, 감지 여부, 로그넘버를 임시저장함
        human_action_detection[label] = {'s_count' : 4, 's_frame' : 0, 'e_count' : 0, 'is_detected' : False, 'log_num' : 0}

    
    # 최종 저장할 이상행동 감지 로그변수
    abnormal_action_log = []









    # 비디오의 첫 프레임 읽기
    logger.debug('video read+')
    video = cv2.VideoCapture(args.video)
    ret_val, frame = video.read()

    # video width 가 너무 크면 속도가 느려져서 일정 제한 이상에서 width와 height를 절반으로 downscaling함
    video_width_limit = 1000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    logger.info('video frame=%dx%d' % (frame.shape[1], frame.shape[0]))
    

    # 모델(의 그래프)을 불러옴
    action_graph = act_class.graph

    # frame 읽기 실패한 횟수
    fail_count = 0

    # 동영상 파일의 초당 프레임 수
    video_fps = 30

    # 경로를 받음
    path = args.video
    # 확장자 자르는 용도로 .단위로 스플릿함 (폴더 이름에 .넣는 경우도 있는데 파일 이름에 확장자표시 말고 추가로 없으면 상관없음)
    # 해당 결과 path는 [(경로들)+동영상이름, 확장자]가 됨. ((경로들)이 절대경로(c:, d:어쩌구)라도 상관없음)
    path = path.split('.')

    # 잘린거에서 백슬래시(\)를 없애기 위해 그걸로 스플릿하여 temp에 넣음
    # 해당 결과 temp는 (경로들)이 \로 이루어졌다면 [폴더명, 폴더명, .. , 동영상이름, 확장자]가 됨
    # (경로들)이 /로 이루어졌다면 [폴더명/폴더명/.../동영상이름, 확장자]가 됨
    temp = []
    for p in path:
        for a in p.split('\\'):
            temp.append(a)
    # 위의 과정과 마찬가지로 슬래시(/) 경로를 처리
    # 해당 결과 path는 무조건 [폴더명, 폴더명, ... , 폴더명, 동영상이름, 확장자]가 됨
    path = []
    for p in temp:
        for a in p.split('/'):
            path.append(a)
    # path의 뒤에서 1번째는 확장자, 2번째는 동영상 이름이므로 2번째를 video_name로 저장
    video_name = path[-2]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if not os.path.exists('output_video'):
        os.makedirs('output_video')
    out = cv2.VideoWriter('output_video\\output_' + video_name + '.avi', fourcc, video_fps, (frame.shape[1], frame.shape[0]), isColor=True)

    # 현재 처리중인 프레임 번호
    processing_frame = 0

    # 처리시간 계산용 변수
    fps_time = time.time()

    # 처리시간따라 다음번에 출력할 프레임을 위해 부분적으로 카운트하는 변수
    frame_count = 0


    # 감지로그 번호
    detection_log_num = 1

    # 검사주기 변수
    detection_cycle = 0

    # 오픈된 동안 프레임을 읽음
    while video.isOpened():
        # 토탈 몇 번째 프레임인지 카운트
        processing_frame += 1
        # 부분적으로 몇 번째 프레임인지 카운트
        frame_count += 1
        
        # 프레임 읽음
        ret_val, frame = video.read()
        

        # 100번 이상 프레임읽기 실패하면 스톱
        if ret_val is False:
            fail_count += 1
            if fail_count > 100:
                break
            continue
        # if time.time() - fps_time > frame_count / video_fps:
        #     continue

        # if processing_frame % 5 != 0:
        #     continue

        # if processing_frame > 250:
        #     break


        '''
        직전 회차에 처리시간동안 흘렀어야할 프레임만큼 건너뜀
        ex) video_fps = 30인 동영상에서
        직전 프레임이 1.5초만에 처리되었다면
        1.5 > frame_count / 30 에서 약 45~46프레임 이후의 프레임을 처리하도록 하여 전체 영상 속도에 영향을 덜 가게 함
        '''
        # if time.time() - fps_time > frame_count / video_fps:
        #     continue

        # 읽은 프레임을 처리하기 시작한 시간 기록
        fps_time = time.time()

        # 처리하기 시작할 때 부분카운트 리셋
        frame_count = 0

        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        # 보여줄 이미지(output_image)를 읽은 frame을 복사하여 변수로 사용
        output_image = frame

        # (현재 프레임 / fps / 60)으로 동영상의 분을, (현재 프레임 / fps % 60) 으로 동영상의 초를 출력화면 좌상단에 표시
        cv2.putText(output_image,
                            "%0d : %0d" %(int(processing_frame / video_fps / 60), int(processing_frame / video_fps % 60)),
                            (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
        
        # TfPoseEstimator에서 PreTrain 된 model을 통해 사람의 skeleton point를 찾아내서 반환
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        '''
        e.inference : 내부 동작은 c++과 연계되어 코드 해석 불가 (F12눌러서 봐도 주석 없음)
        '''

        # skeleton point를 통해 한 프레임에서 사람들의 범위를 상하좌우 픽셀단위로 변환함.
        boundarys = TfPoseEstimator.get_humans_imgbox(frame, humans, 0.02)
        '''
        직접 만든 함수 get_humans_imgbox
        '''

        
        # 각 사람별로 이상행동이 감지되었는지 저장
        # 이때 중복할 필요는 없어서 set형태로 저장
        predict_action_list = set()





        # boundarys : {사람1 : [left, right, up, down], 사람2 : [left, right, up, down], ... }
        for num_of_human in boundarys:
            print("현재 프레임 넘버 : ", processing_frame)
            print("human_action_detection")
            for a in human_action_detection:
                print(a, " : ", human_action_detection[a])
            print("abnormal_action_log")
            for a in abnormal_action_log:
                print(a)
            # 각 변수들을 더 직관적으로 임시저장
            img_left = boundarys[num_of_human][0]
            img_right = boundarys[num_of_human][1]
            img_up = boundarys[num_of_human][2]
            img_down = boundarys[num_of_human][3]
            
            # 프레임 크기와 동일한 흰 바탕의 이미지를 그린다.
            sub_img_ske = np.zeros(frame.shape,dtype=np.uint8)
            sub_img_ske.fill(255)

            # 해당 이미지 위에다가 num_of_human 번째 사람의 skeleton 이미지를 그린다.
            sub_img_ske = TfPoseEstimator.draw_humans(sub_img_ske, [humans[num_of_human]], imgcopy=False)
            '''
            draw_humans
            '''

            # 흰 바탕의 skeleton이미지로 어떤 액션인지(normal / abnormal) 판정한다.
            # 단, 판정 때 앞서 추출한 이미지의 해당 사람 부분만을 넣어 판정한다.
            predict_action = act_class.classify(sub_img_ske[img_up:img_down, img_left:img_right], graph=action_graph)


            # 각 사람별 탐지된 이상행동을 set에 넣는다.
            predict_action_list.add(predict_action['predict_action'])




            '''
            classify : CNN기반 mobilenet 모델로 각 사람에 대한 판정, 내부 코드 알기는 힘듦
            '''

            # 판정한 내용에 따라 보여줄 이미지(output_image)에 표시한다.
            cv2.putText(output_image,
                        "%s, %f" %(predict_action['predict_action'], predict_action['predict_score']),
                        (img_left, img_up + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.rectangle(output_image, (img_left, img_up), (img_right, img_down), (0, 0, 255))
            
            # 찾아낸 skeleton point를 output image에 점과 연결선으로 표시
            # 출력이미지에도 스켈레톤 이미지를 그리는 것인데, 그리면 복잡해져서 안 그리는게 더 나아보인다.
            # output_image = TfPoseEstimator.draw_humans(frame, [humans[num_of_human]], imgcopy=False)
        
        out.write(output_image)

        # 처리한 한 프레임에 대해 이미지를 출력한다.
        # 이 때, output_image[상:하, 좌:우]를 (픽셀단위로) 하면 원하는 부분만 출력할 수도 있다.
        cv2.imshow('tf-pose-estimation result', output_image)
        
        predict_action_list.discard('0001')
        
        for abaction in predict_action_list:
            if human_action_detection[abaction]['s_count'] > 0:
                human_action_detection[abaction]['s_count'] -= 1
            
            if human_action_detection[abaction]['is_detected'] == True and human_action_detection[abaction]['e_count'] > 0:
                human_action_detection[abaction]['e_count'] -= 1
            
            if human_action_detection[abaction]['s_count'] == 2 and human_action_detection[abaction]['is_detected'] == False:
                human_action_detection[abaction]['s_frame'] = processing_frame
            
            if human_action_detection[abaction]['s_count'] == 0 and human_action_detection[abaction]['is_detected'] == False:
                human_action_detection[abaction]['is_detected'] = True
                human_action_detection[abaction]['log_num'] = detection_log_num
                abnormal_action_log.append([detection_log_num, abaction, human_action_detection[abaction]['s_frame']])
                detection_log_num += 1

        detection_cycle += 1
        if detection_cycle % 2 == 0:
            detection_cycle = 0
            
            for abaction in human_action_detection:
                abaction_dict = human_action_detection[abaction]
                if abaction_dict['is_detected'] == False and abaction_dict['s_count'] < 4:
                    abaction_dict['s_count'] += 1
                if abaction_dict['is_detected'] == True:
                    abaction_dict['e_count'] += 1
                if abaction_dict['e_count'] >= 5:
                    log_num = abaction_dict['log_num']
                    abnormal_action_log[log_num - 1].append(processing_frame)

                    abaction_dict['s_count'] = 4
                    abaction_dict['s_frame'] = 0
                    abaction_dict['e_count'] = 0
                    abaction_dict['is_detected'] = False
                    abaction_dict['log_num'] = 0

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for abaction in human_action_detection:
        abaction_dict = human_action_detection[abaction]
        if abaction_dict['is_detected'] == True:
            log_num = abaction_dict['log_num']
            abnormal_action_log[log_num - 1].append(processing_frame)
            abaction_dict['is_detected'] = False
    video.release()

    cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python predict.py --video=testvideo.mp4
# python predict.py --video=test_video/488-1_cam01_vandalism01_place09_day_spring_1.mp4
# =============================================================================
