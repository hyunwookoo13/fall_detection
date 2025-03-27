import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (경량 모델 사용)
model = YOLO('yolov8n.pt')

# MacBook 카메라 스트림 열기
cap = cv2.VideoCapture(0)

# 연속된 프레임에서 fall 상태가 지속되는 횟수를 체크하기 위한 변수
fall_counter = 0
# fall 상태로 판단할 연속 프레임 임계치 (예: 3 프레임 이상 지속되면 낙상으로 간주)
fall_threshold = 3

def check_posture(bbox, ratio_threshold=1.0):
    """
    bbox: (x1, y1, x2, y2) 형식의 bounding box
    ratio_threshold: 낙상 판단을 위한 width/height 임계치
    width가 height보다 크면(즉, height가 작으면) 낙상으로 판단
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h <= 0:
        return False
    aspect_ratio = w / h
    return aspect_ratio > ratio_threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라로부터 프레임을 읽지 못했습니다.")
        break

    # YOLO 모델로 객체 검출 수행
    results = model(frame)[0]
    
    # 해당 프레임에 낙상 상태가 감지되었는지 여부
    fall_detected_frame = False
    
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        # COCO 데이터셋 기준, 사람은 클래스 0번입니다.
        if int(cls) == 0:
            bbox = (x1, y1, x2, y2)
            if check_posture(bbox, ratio_threshold=1.0):
                fall_detected_frame = True
                color = (0, 0, 255)  # 빨간색: 낙상
                label = "Fall Detected"
            else:
                color = (0, 255, 0)  # 초록색: 서 있음
                label = "Standing"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # 연속된 프레임에서 낙상 상태가 지속되는지 체크
    if fall_detected_frame:
        fall_counter += 1
    else:
        fall_counter = 0  # 조건에 맞지 않는 프레임이 나오면 카운터 리셋

    # 일정 프레임 이상 낙상 상태가 지속되면 경고 출력
    if fall_counter >= fall_threshold:
        cv2.putText(frame, "ALERT: Fall!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
