import threading
import queue
import ffmpeg
import numpy as np
import cv2
import time


FRAME_WIDTH = 384
FRAME_HEIGHT = 384

# Khởi tạo queue để chia sẻ frame giữa các luồng
frame_queue = queue.Queue(maxsize=10)

def rtsp_reader(cam_source, frame_queue):
    args = {"rtsp_transport": "udp"}
    probe = ffmpeg.probe(cam_source)
    cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
    width = cap_info["width"]
    height = cap_info["height"]
    process = (
        ffmpeg.input(cam_source, **args)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )

    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame_rgb = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])  # RGB

        # Resize frame RGB
        resized_rgb = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT))

        # Chuyển sang BGR để OpenCV xử lý, hiển thị
        bgr_frame = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

        if not frame_queue.full():
            # Đưa vào queue: (frame gốc RGB không resize, frame BGR resize dùng xử lý)
            frame_queue.put((frame_rgb.copy(), bgr_frame))

        if cv2.waitKey(1) == ord("q"):
            break
    process.kill()

def main_detection(frame_queue):
    import torch
    from Detection.Utils import ResizePadding
    from DetectorLoader import TinyYOLOv8_onecls
    from PoseEstimateLoader import SPPE_FastPose
    from Track.Tracker import Detection, Tracker
    from ActionsEstLoader import TSSTG
    from fn import draw_single

    device = 'cuda'
    inp_dets = 384
    inp_pose = (224, 160)
    detect_model = TinyYOLOv8_onecls(inp_dets, device=device)
    pose_model = SPPE_FastPose('resnet50', *inp_pose, device=device)
    tracker = Tracker(max_age=45, n_init=3)
    action_model = TSSTG()
    resize_fn = ResizePadding(inp_dets, inp_dets)

    def kpt2bbox(kpt, ex=20):
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                         kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

    fps_time = time.time()
    while True:
        if not frame_queue.empty():
            original_rgb, bgr_frame = frame_queue.get()

            # Khi đưa vào model, chuyển BGR sang RGB
            rgb_for_model = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            detected = detect_model.detect(rgb_for_model, need_resize=False, expand_bb=10)

            if detected is not None:
                filtered = []
                h_img, w_img = bgr_frame.shape[:2]
                for det in detected:
                    x1, y1, x2, y2, conf, class_id = det
                    if class_id == 0 and conf >= 0.9:
                        filtered.append([x1, y1, x2, y2, conf, class_id])
                detected = torch.tensor(filtered, dtype=torch.float32) if filtered else None

            tracker.predict()
            detections = []

            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            if detected is not None:
                poses = pose_model.predict(rgb_for_model, detected[:, 0:4], detected[:, 4])
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(), ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)
                action = 'pending..'
                clr = (0, 255, 0)

                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, bgr_frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name == 'Fall Down':
                        clr = (0, 0, 255)
                    elif action_name == 'Lying Down':
                        clr = (0, 200, 255)

                if track.time_since_update == 0:
                    bgr_frame = draw_single(bgr_frame, track.keypoints_list[-1])
                    bgr_frame = cv2.rectangle(bgr_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 1)
                    bgr_frame = cv2.putText(bgr_frame, str(track_id), center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
                    bgr_frame = cv2.putText(bgr_frame, action, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, clr, 1)

            bgr_frame = cv2.putText(bgr_frame, 'FPS: {:.2f}'.format(1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_time = time.time()

            # Hiển thị:
            resized_original = cv2.resize(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), (384, 384))
            resized_detection = cv2.resize(bgr_frame, (768, 768))

            cv2.imshow("Original Camera", resized_original)
            cv2.imshow("Fall Detection", resized_detection)

            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == '__main__':
    cam_source = "rtsp://admin:usename@ip address"
    t1 = threading.Thread(target=rtsp_reader, args=(cam_source, frame_queue))
    t2 = threading.Thread(target=main_detection, args=(frame_queue,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
