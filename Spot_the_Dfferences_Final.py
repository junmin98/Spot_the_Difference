# 사용할 라이브러리 사용
import numpy as np
import cv2
import threading
import time

# 전역 변수 선언
level = 1           # level 초기값 = 1
R = (0, 0, 255)
diff = 0            # 마우스로 선택한 그림 영역이 같은지, 다른지를 판단할 변수
num_of_diff = 0     # 다른 그림을 찾은 개수
count = 0           # 타이머에 사용할 변수
# flags
flag_time = True    # 타이머 시작/종료에 관한 flag
start = True        # 게임 시작 flag
# etc
font = cv2.FONT_HERSHEY_COMPLEX     # 사용할 폰트

# LEVEL1 -> img1,2이용 / LEVEL2 -> img3,4이용
# 이미지를 받아오고, 게임에 사용하기 위한 image processing
img1 = cv2.imread('room_1.JPG')
img2 = cv2.imread('room_2.JPG')
img3 = cv2.imread('boat_1.JPG')
img4 = cv2.imread('boat_2.JPG')
# image resize
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))
img3 = cv2.resize(img3, (600, 400))
img4 = cv2.resize(img4, (600, 400))

# ######### COLOR IMAGE 를 이용할 경우, 잡음 제거와 Thresholding 하는 부분은 생략 가능 ##########
# """
# 이미지의 잡음 제거
img1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
img2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)
img3 = cv2.fastNlMeansDenoisingColored(img3, None, 10, 10, 7, 21)
img4 = cv2.fastNlMeansDenoisingColored(img4, None, 10, 10, 7, 21)
# image threshold -> binary image
ret, img1 = cv2.threshold(img1, 180, 255, cv2.THRESH_BINARY)
ret, img2 = cv2.threshold(img2, 180, 255, cv2.THRESH_BINARY)
ret, img3 = cv2.threshold(img3, 180, 255, cv2.THRESH_BINARY)
ret, img4 = cv2.threshold(img4, 180, 255, cv2.THRESH_BINARY)
# """
###########################################################################################

# 이미지를 이용해 연산을 하기 위해 배열의 형태로 바꾼다.
img1 = np.array(img1, dtype=np.uint8)
img2 = np.array(img2, dtype=np.uint8)
img3 = np.array(img3, dtype=np.uint8)
img4 = np.array(img4, dtype=np.uint8)

##########################################################################################

# LEVEL 1의 경우 틀린 부분 : 6군데 / LEVEL 2의 경우 10군데 존재

# 이미지에서 마우스로 선택한 위치의 좌표 값을 받아오는 함수
def onMouse(event, x, y, flags, param):
    # 사용할 전역 변수 선언
    global num_of_diff, diff
    # setMouseCallback 함수로부터 받아온 parameter(이미지)
    param1 = param[0]   # 원본 이미지
    param2 = param[1]   # 원본 이미지와 비교하여 다른 그림을 찾을 이미지

    # 마우스 왼쪽 버튼을 누르면, 선택한 위치의 좌표 값을 받아온다.
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("x,y:", x, y)
        # 좌표값을 중심으로 두 이미지의 7x7 내에 존재하는 픽셀 값을 roi_1과 roi_2에 각각 저장
        roi_1 = param1[y-3:y+4, x-3:x+4]
        roi_2 = param2[y-3:y+4, x-3:x+4]
        # Canny detector 를 이용해 roi_1과 roi_2에 존재하는 sub image 의 edge 를 detect
        edge1 = cv2.Canny(roi_1, 170, 200)
        edge2 = cv2.Canny(roi_2, 170, 200)
        # cv2.imshow('edge1', edge1)
        # cv2.imshow('edge2', edge2)
        sum1 = np.sum(edge1)
        sum2 = np.sum(edge2)
        # 두 sub image의 픽셀 값들을 각각 합한 후 둘의 차이, diff를 계산
        diff = np.abs(sum1 - sum2)

        # edge1과 edge2의 요소들이 각각 49개 중, 5개 이상이 다르다면, 그 영역은 다른 그림이라고 생각한다.
        # 즉, diff 의 크기가 255*5보다 크다면 선택한 위치의 그림은 다른 것이다.
        # 두 그림이 다르다면, 좌표 값을 중심으로, 이미지에 동그라미를 그려 다른 부분을 나타낸다.
        if diff >= 5*255:
            cv2.circle(param2, (x, y), 20, R, 3)
            count_the_num_of_diff()  # 다른 그림의 개수를 센다.
            cv2.namedWindow('Original Image')
            cv2.namedWindow('Spot the Difference')
            cv2.imshow('Original Image', param1)
            cv2.imshow('Spot the Difference', param2)


# 사용자가 찾은 다른 부부을 개수를 세주는 함수
def count_the_num_of_diff():
    # 사용할 전역변수 선언
    global num_of_diff
    # 이 함수가 실행되면, 다른 그림을 찾은 개수를 센다.
    num_of_diff = num_of_diff + 1
    # print("num of diff :", num_of_diff)


# 게임 시간에 제한을 두기 위한 타이머 함수 구현.
def start_timer():
    global count, start, num_of_diff, flag_time, R, level
    count += 1
    timer = threading.Timer(1, start_timer)
    timer.start()
    # print('time:', count)

    # 'Esc' 버튼이 눌리면(->flag_time == False), 타이머는 종료된다.
    if not flag_time:
        timer.cancel()

    # 남은 시간을 알려주는 문구 출력.
    if 14 < count < 16:
        print("45 seconds left")
    if 29 < count < 31:
        print("30 seconds left")
    if 44 < count < 46:
        print("15 seconds left")
    if 54 < count < 56:
        print("5 seconds left")
    if 56 < count < 59:
        print("The game will end soon.")

    # LEVEL1 : 게임 시간이 지났는데, 다른 그림을 다 찾지 못했을 경우
    if (count > 60) and (num_of_diff < 6):
        cv2.putText(img1, 'GAME OVER!', (150, 250), font, 1, R, 2)
        timer.cancel()     # game 종료 후, timer 종료한다
        cv2.waitKey(5000)  # GAME OVER 후 5초 뒤에 게임 창이 닫힌다.
        start = False
        cv2.destroyAllWindows()

    # LEVEL1 : 게임 시간(60초) 안에, 다른 그림을 모두 찾았을 경우
    if (5 < count < 60) and (num_of_diff == 6):
        cv2.putText(img1, 'GAME CLEAR!', (150, 250), font, 1, R, 2)
        time.sleep(2)
        level = 2   # LEVEL UP
        print(" LEVEL 2, find the 10 differences ")
        count = 0   # count 다시 0으로 초기화
        cv2.waitKey(3000)   # 3초 뒤 다음 레벨의 사진이 뜬다
        exit()

    if level == 2:
        # LEVEL 2 : 60초 안에 틀린 그림을 다 못찾았을 경우
        if (count > 60) and (5 <= num_of_diff < 16):
            cv2.putText(img3, 'GAME OVER!', (200, 200), font, 1, R, 2)
            timer.cancel()      # game 종료 후, timer 종료
            cv2.waitKey(5000)   # GAME OVER 후 5초 뒤에 게임 창이 닫힌다.
            start = False       # GAME 종료
            cv2.destroyAllWindows()
        # LEVEL 2 : 60초 안에 틀린 그림을 모두 찾았을 경우, GAME CLEAR 문구를 띄운 후, 게임을 종료한다.
        if (count < 60) and (num_of_diff == 16):
            cv2.putText(img3, 'GAME CLEAR!', (200, 200), font, 1, R, 2)
            timer.cancel()
            print("Excellent! The game is complete. ")
            cv2.waitKey(5000)
            start = False
            cv2.destroyAllWindows()
            exit()


# 게임 도중 힌트 사용을 위한 함수 : 두 영상의 차이를 비교하여, 정답을 2초간 보여준다.
def make_hint():
    if level == 1:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff_btw_img_1 = cv2.subtract(img1_gray, img2_gray)
        diff_btw_img_2 = cv2.subtract(img1_gray, img2_gray)
        diff_btw_img_level1 = diff_btw_img_1 + diff_btw_img_2
        cv2.namedWindow('HINT')
        cv2.imshow('HINT', diff_btw_img_level1)
        # Hint 사진을 2초간 보여주고 창을 닫는다.
        cv2.waitKey(2000)
        cv2.destroyWindow('HINT')

    if level == 2:
        img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        diff_btw_img_3 = cv2.subtract(img3_gray, img4_gray)
        diff_btw_img_4 = cv2.subtract(img4_gray, img3_gray)
        diff_btw_img_level2 = diff_btw_img_3 + diff_btw_img_4
        cv2.namedWindow('HINT')
        cv2.imshow('HINT', diff_btw_img_level2)
        # Hint 사진을 2초간 보여주고 창을 닫는다.
        cv2.waitKey(2000)
        cv2.destroyWindow('HINT')


# 게임 안내 메세지 출력
print("Spot the Difference ! Time Limit : 60sec")
print("If you want to get the Hint, Press the 'h'")
print("If you want to finish the game, press 'esc' ")
print(" LEVEL 1, find the 6 differences")
start_timer()


while start:
    if level == 1:
        cv2.namedWindow('Original Image')
        cv2.namedWindow('Spot the Difference')
        cv2.imshow('Original Image', img1)
        cv2.imshow('Spot the Difference', img2)
        cv2.setMouseCallback('Spot the Difference', onMouse, param=[img1, img2])

    if level == 2:
        cv2.namedWindow('Original Image')
        cv2.namedWindow('Spot the Difference')
        cv2.imshow('Original Image', img3)
        cv2.imshow('Spot the Difference', img4)
        cv2.setMouseCallback('Spot the Difference', onMouse, param=[img3, img4])

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # 'Esc' 를 누르면, 게임과 타이머가 종료된다.
        cv2.destroyAllWindows()
        flag_time = False
        break

    if k == ord('h'):  # 'h'를 누르면, 약 2초동안 힌트 이미지를 볼 수 있다.
        make_hint()


