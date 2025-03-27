import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (경량 모델 사용)
model = YOLO('yolov8n.pt')

# MacBook 카메라 스트림 열기
cap = cv2.VideoCapture(0)

# 연속된 프레임에서 낙상 상태가 지속되는 횟수를 체크하기 위한 변수
fall_counter = 0
# 연속된 프레임 중 낙상 상태로 판단할 임계치 (3 프레임 이상이면 ALERT)
fall_threshold = 5

def check_posture(bbox):
    """
    bbox: (x1, y1, x2, y2) 형식의 bounding box
    가로/세로 비율(aspect_ratio)에 따라
      - 1.0 미만 : Standing
      - 1.0 이상 ~ 2.0 미만 : Sitting
      - 2.0 이상 : Fall Detected
    로 분류합니다.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h <= 0:
        return "Unknown"  # 높이가 0 이하이면 잘못된 값

    aspect_ratio = w / h

    if aspect_ratio < 1.0:
        return "normal"
    elif aspect_ratio < 2.0:
        return "normal"
    else:
        return "Fall Detected"

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라로부터 프레임을 읽지 못했습니다.")
        break

    # YOLO 모델로 객체 검출 수행
    results = model(frame)[0]
    
    # 현재 프레임에서 낙상 상태를 감지했는지 여부
    fall_detected_frame = False

    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        # COCO 데이터셋 기준, 사람은 클래스 0번
        if int(cls) == 0:
            bbox = (x1, y1, x2, y2)
            posture = check_posture(bbox)

            # 상태별 색상 지정
            if posture == "Fall Detected":
                color = (0, 0, 255)  # 빨간색
                fall_detected_frame = True
            elif posture == "Sitting":
                color = (0, 255, 255)  # 노란색
            else:
                color = (0, 255, 0)  # 초록색

            # bounding box 및 상태 라벨 표시
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, posture, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # 연속된 프레임에서 낙상 상태가 지속되는지 확인
    if fall_detected_frame:
        fall_counter += 1
    else:
        fall_counter = 0

    # 일정 프레임 이상 낙상 상태가 지속되면 ALERT 메시지 표시
    if fall_counter >= fall_threshold:
        cv2.putText(frame, "ALERT: Fall!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
