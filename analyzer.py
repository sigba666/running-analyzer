"""
analyzer.py – RunningAnalyzer v4 (mediapipe Tasks API, >= 0.10.30 호환)
"""
import cv2, math, os, urllib.request
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 포즈 랜드마크 모델 자동 다운로드
MODEL_PATH = "pose_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_lite/float16/latest/"
              "pose_landmarker_lite.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 랜드마크 인덱스 (BlazePose 33점 동일)
class _PL:
    NOSE           = 0
    LEFT_SHOULDER  = 11; RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13; RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15; RIGHT_WRIST    = 16
    LEFT_HIP       = 23; RIGHT_HIP      = 24
    LEFT_KNEE      = 25; RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27; RIGHT_ANKLE    = 28

_R = dict(shoulder=_PL.RIGHT_SHOULDER, hip=_PL.RIGHT_HIP,
          knee=_PL.RIGHT_KNEE, ankle=_PL.RIGHT_ANKLE,
          elbow=_PL.RIGHT_ELBOW, wrist=_PL.RIGHT_WRIST)
_L = dict(shoulder=_PL.LEFT_SHOULDER, hip=_PL.LEFT_HIP,
          knee=_PL.LEFT_KNEE, ankle=_PL.LEFT_ANKLE,
          elbow=_PL.LEFT_ELBOW, wrist=_PL.LEFT_WRIST)

# 스켈레톤 연결선
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
]

SPEED_PRESETS = {
    'slow':   dict(knee=155.0, trunk=7.0, cadence=165.0, v_osc=7.0),
    'normal': dict(knee=160.0, trunk=5.0, cadence=175.0, v_osc=8.0),
    'fast':   dict(knee=163.0, trunk=4.0, cadence=180.0, v_osc=9.0),
}
STABILITY_THRESHOLD = 20.0
ASYMMETRY_THRESHOLD = 15.0


def _lm(lms, idx):  return (lms[idx].x, lms[idx].y)

def _angle_3pts(a, b, c):
    ba=(a[0]-b[0],a[1]-b[1]); bc=(c[0]-b[0],c[1]-b[1])
    cos=(ba[0]*bc[0]+ba[1]*bc[1])/(math.hypot(*ba)*math.hypot(*bc)+1e-9)
    return math.degrees(math.acos(max(-1.,min(1.,cos))))

def _trunk_angle(shoulder, hip):
    return math.degrees(math.atan2(abs(hip[0]-shoulder[0]),abs(hip[1]-shoulder[1])))

def _safe_savgol(arr, win=15, poly=2):
    n=len(arr)
    if n<5: return np.array(arr)
    w=min(win,n if n%2==1 else n-1)
    if w<3: w=3
    if w%2==0: w-=1
    return savgol_filter(arr,w,poly)

def _score_knee(e,ideal):    return max(0,round(100-e*2.5))
def _score_trunk(e,ideal):   return max(0,round(100-e*5))
def _score_cadence(c,ideal):
    if abs(c-ideal)<=5: return 100
    return max(0,round(100-abs(c-ideal)*2))
def _score_v_osc(v,ideal):
    if abs(v-ideal)<=2: return 100
    return max(0,round(100-abs(v-ideal)*10))
def _score_symmetry(a):  return max(0,round(100-a*3))
def _score_arm_swing(a): return max(0,round(100-a*4))


def _create_landmarker(running_mode):
    _ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def analyze_video(video_path:str, height_cm:float=170.0,
                  speed_level:str='normal', progress_cb=None) -> dict:

    preset = SPEED_PRESETS.get(speed_level, SPEED_PRESETS['normal'])
    IDEAL_KNEE    = preset['knee']
    IDEAL_TRUNK   = preset['trunk']
    IDEAL_CADENCE = preset['cadence']
    IDEAL_V_OSC   = preset['v_osc']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames=[]; r_knees=[]; l_knees=[]; r_trunks=[]; l_trunks=[]
    knee_asym=[]; trunk_asym=[]; frame_badness=[]
    r_ankle_y=[]; l_ankle_y=[]; hip_mid_y=[]; sho_mid_y=[]; ank_mid_y=[]
    r_arm_angles=[]; l_arm_angles=[]; arm_asym=[]

    with _create_landmarker(mp_vision.RunningMode.VIDEO) as landmarker:
        for fi in range(total):
            ret,frame=cap.read()
            if not ret: break
            if progress_cb: progress_cb(fi+1,total)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(fi*1000/fps))

            if not res.pose_landmarks: continue
            lms = res.pose_landmarks[0]

            rk=_angle_3pts(_lm(lms,_R['hip']),_lm(lms,_R['knee']),_lm(lms,_R['ankle']))
            lk=_angle_3pts(_lm(lms,_L['hip']),_lm(lms,_L['knee']),_lm(lms,_L['ankle']))
            rt=_trunk_angle(_lm(lms,_R['shoulder']),_lm(lms,_R['hip']))
            lt=_trunk_angle(_lm(lms,_L['shoulder']),_lm(lms,_L['hip']))
            ra=_angle_3pts(_lm(lms,_R['shoulder']),_lm(lms,_R['elbow']),_lm(lms,_R['wrist']))
            la=_angle_3pts(_lm(lms,_L['shoulder']),_lm(lms,_L['elbow']),_lm(lms,_L['wrist']))

            r_ay=lms[_R['ankle']].y; l_ay=lms[_L['ankle']].y
            r_hy=lms[_R['hip']].y;  l_hy=lms[_L['hip']].y
            r_sy=lms[_R['shoulder']].y; l_sy=lms[_L['shoulder']].y

            knee_e=(abs(rk-IDEAL_KNEE)+abs(lk-IDEAL_KNEE))/2
            trunk_e=(abs(rt-IDEAL_TRUNK)+abs(lt-IDEAL_TRUNK))/2
            asym_e=abs(rk-lk)
            badness=0.45*(knee_e/40)+0.3*(trunk_e/20)+0.25*(asym_e/30)

            frames.append(fi)
            r_knees.append(rk); l_knees.append(lk)
            r_trunks.append(rt); l_trunks.append(lt)
            knee_asym.append(abs(rk-lk)); trunk_asym.append(abs(rt-lt))
            r_ankle_y.append(r_ay); l_ankle_y.append(l_ay)
            hip_mid_y.append((r_hy+l_hy)/2)
            sho_mid_y.append((r_sy+l_sy)/2)
            ank_mid_y.append((r_ay+l_ay)/2)
            frame_badness.append(round(float(badness),4))
            r_arm_angles.append(ra); l_arm_angles.append(la)
            arm_asym.append(abs(ra-la))

    cap.release()
    if not frames: return {"error":"포즈를 감지하지 못했습니다."}

    rk_a=np.array(r_knees); lk_a=np.array(l_knees)
    rt_a=np.array(r_trunks); lt_a=np.array(l_trunks)
    ka_a=np.array(knee_asym); ta_a=np.array(trunk_asym)
    ra_a=np.array(r_arm_angles); la_a=np.array(l_arm_angles)
    aa_a=np.array(arm_asym); bd_a=np.array(frame_badness)

    duration_sec=len(frames)/fps
    min_dist=max(int(fps*0.25),3)
    r_peaks,_=find_peaks(_safe_savgol(r_ankle_y),distance=min_dist)
    l_peaks,_=find_peaks(_safe_savgol(l_ankle_y),distance=min_dist)
    cadence=round((len(r_peaks)+len(l_peaks))/duration_sec*60,1) if duration_sec>0 else 0

    hip_s=_safe_savgol(hip_mid_y)
    osc_norm=np.percentile(hip_s,95)-np.percentile(hip_s,5)
    body_h=float(np.mean(np.abs(np.array(ank_mid_y)-np.array(sho_mid_y))))
    v_osc_cm=round(float((osc_norm/body_h)*height_cm),1) if body_h>1e-6 else 0.0

    rk_err=float(np.abs(rk_a-IDEAL_KNEE).mean())
    lk_err=float(np.abs(lk_a-IDEAL_KNEE).mean())
    rt_err=float(np.abs(rt_a-IDEAL_TRUNK).mean())
    lt_err=float(np.abs(lt_a-IDEAL_TRUNK).mean())
    avg_knee_err=(rk_err+lk_err)/2
    avg_trunk_err=(rt_err+lt_err)/2

    scores=dict(
        knee     =_score_knee(avg_knee_err,IDEAL_KNEE),
        trunk    =_score_trunk(avg_trunk_err,IDEAL_TRUNK),
        cadence  =_score_cadence(cadence,IDEAL_CADENCE),
        v_osc    =_score_v_osc(v_osc_cm,IDEAL_V_OSC),
        symmetry =_score_symmetry(float(ka_a.mean())),
        arm_swing=_score_arm_swing(float(aa_a.mean())),
    )

    tips=[]
    if avg_knee_err>15: tips.append("무릎 신전 부족 → 보폭 줄이고 착지 자세 점검")
    if avg_trunk_err>10: tips.append("허리 과도 기울어짐 → 코어 강화 및 상체 수직 유지")
    if cadence<IDEAL_CADENCE-15: tips.append(f"케이던스 낮음({cadence}spm) → 보폭 줄이고 회전수 늘리기")
    elif cadence>IDEAL_CADENCE+25: tips.append(f"케이던스 높음({cadence}spm) → 페이스 조절 필요")
    if v_osc_cm>12: tips.append(f"수직 진동 큼({v_osc_cm}cm) → 상하 바운싱 줄이기")
    elif v_osc_cm<4: tips.append(f"수직 진동 작음({v_osc_cm}cm) → 보폭이 너무 짧을 수 있음")
    if ka_a.mean()>ASYMMETRY_THRESHOLD: tips.append("좌우 무릎 비대칭 → 균형 훈련 필요")
    if aa_a.mean()>20: tips.append("좌우 팔 스윙 비대칭 → 팔 스윙 의식적으로 균형 맞추기")
    if rk_a.std()>STABILITY_THRESHOLD or lk_a.std()>STABILITY_THRESHOLD:
        tips.append("무릎 각도 변동 큼 → 균일한 보폭과 페이스 유지")

    step=max(1,len(frames)//300)
    bd_norm=(bd_a/bd_a.max()).tolist() if bd_a.max()>0 else bd_a.tolist()

    return dict(
        total_frames=total, valid_frames=len(frames),
        fps=fps, duration_sec=round(duration_sec,1),
        height_cm=height_cm, speed_level=speed_level,
        ideal=dict(knee=IDEAL_KNEE,trunk=IDEAL_TRUNK,
                   cadence=IDEAL_CADENCE,v_osc=IDEAL_V_OSC),
        r_knee_mean=round(float(rk_a.mean()),1), r_knee_std=round(float(rk_a.std()),1),
        l_knee_mean=round(float(lk_a.mean()),1), l_knee_std=round(float(lk_a.std()),1),
        r_knee_error_mean=round(rk_err,1), l_knee_error_mean=round(lk_err,1),
        r_trunk_mean=round(float(rt_a.mean()),1), r_trunk_std=round(float(rt_a.std()),1),
        l_trunk_mean=round(float(lt_a.mean()),1), l_trunk_std=round(float(lt_a.std()),1),
        r_trunk_error_mean=round(rt_err,1), l_trunk_error_mean=round(lt_err,1),
        knee_asymmetry_mean=round(float(ka_a.mean()),1),
        trunk_asymmetry_mean=round(float(ta_a.mean()),1),
        knee_asymmetric=bool(ka_a.mean()>ASYMMETRY_THRESHOLD),
        trunk_asymmetric=bool(ta_a.mean()>ASYMMETRY_THRESHOLD),
        r_knee_unstable=bool(rk_a.std()>STABILITY_THRESHOLD),
        l_knee_unstable=bool(lk_a.std()>STABILITY_THRESHOLD),
        cadence=cadence, v_osc_cm=v_osc_cm,
        arm_swing_asymmetry=round(float(aa_a.mean()),1),
        r_arm_mean=round(float(ra_a.mean()),1),
        l_arm_mean=round(float(la_a.mean()),1),
        scores=scores, tips=tips,
        chart=dict(
            frames    =frames[::step],
            r_knee    =[round(v,1) for v in r_knees[::step]],
            l_knee    =[round(v,1) for v in l_knees[::step]],
            r_trunk   =[round(v,1) for v in r_trunks[::step]],
            l_trunk   =[round(v,1) for v in l_trunks[::step]],
            knee_asym =[round(v,1) for v in list(ka_a)[::step]],
            r_arm     =[round(v,1) for v in r_arm_angles[::step]],
            l_arm     =[round(v,1) for v in l_arm_angles[::step]],
            arm_asym  =[round(v,1) for v in list(aa_a)[::step]],
            badness   =[round(v,3) for v in bd_norm[::step]],
            r_strikes =[frames[i] for i in r_peaks if i<len(frames)],
            l_strikes =[frames[i] for i in l_peaks if i<len(frames)],
        ),
    )


def create_overlay_video(video_path:str, output_path:str,
                         result:dict, progress_cb=None):
    """분석 결과를 영상에 스켈레톤·각도·점수로 오버레이한 mp4 생성"""
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps  =cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_path,fourcc,fps,(W,H))

    sc=result.get('scores',{})
    overall=round(sum(sc.values())/len(sc)) if sc else 0
    ideal=result.get('ideal',{})

    def _color_score(val, ideal_val, tol=15):
        err=abs(val-ideal_val)
        if err<tol*0.5: return (50,200,50)
        if err<tol:     return (50,200,220)
        return (50,80,220)

    with _create_landmarker(mp_vision.RunningMode.VIDEO) as landmarker:
        for fi in range(total):
            ret,frame=cap.read()
            if not ret: break
            if progress_cb: progress_cb(fi+1,total)

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(fi*1000/fps))

            if res.pose_landmarks:
                lms = res.pose_landmarks[0]

                # 스켈레톤 그리기
                for s,e in POSE_CONNECTIONS:
                    if s<len(lms) and e<len(lms):
                        cv2.line(frame,
                            (int(lms[s].x*W),int(lms[s].y*H)),
                            (int(lms[e].x*W),int(lms[e].y*H)),
                            (180,180,180),2)
                for idx in range(min(33,len(lms))):
                    cv2.circle(frame,(int(lms[idx].x*W),int(lms[idx].y*H)),3,(255,255,255),-1)

                def pt(idx): return (int(lms[idx].x*W),int(lms[idx].y*H))

                rk=_angle_3pts((lms[_R['hip']].x,lms[_R['hip']].y),
                               (lms[_R['knee']].x,lms[_R['knee']].y),
                               (lms[_R['ankle']].x,lms[_R['ankle']].y))
                lk=_angle_3pts((lms[_L['hip']].x,lms[_L['hip']].y),
                               (lms[_L['knee']].x,lms[_L['knee']].y),
                               (lms[_L['ankle']].x,lms[_L['ankle']].y))
                rt=_trunk_angle((lms[_R['shoulder']].x,lms[_R['shoulder']].y),
                                (lms[_R['hip']].x,lms[_R['hip']].y))

                rk_pt=pt(_R['knee']); lk_pt=pt(_L['knee'])
                rk_col=_color_score(rk,ideal.get('knee',160))
                lk_col=_color_score(lk,ideal.get('knee',160))
                cv2.circle(frame,rk_pt,8,rk_col,-1)
                cv2.circle(frame,lk_pt,8,lk_col,-1)
                cv2.putText(frame,f"R:{rk:.0f}",(rk_pt[0]+10,rk_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,rk_col,2,cv2.LINE_AA)
                cv2.putText(frame,f"L:{lk:.0f}",(lk_pt[0]-60,lk_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,lk_col,2,cv2.LINE_AA)

                hip_pt=pt(_R['hip'])
                tr_col=_color_score(rt,ideal.get('trunk',5),tol=10)
                cv2.putText(frame,f"T:{rt:.0f}",(hip_pt[0]+10,hip_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,tr_col,2,cv2.LINE_AA)

            overlay=frame.copy()
            cv2.rectangle(overlay,(0,0),(W,52),(0,0,0),-1)
            cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
            cv2.putText(frame,f"Frame {fi+1}/{total}",
                (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1,cv2.LINE_AA)
            sc_col=(50,200,50) if overall>=80 else (50,200,220) if overall>=55 else (50,80,220)
            cv2.putText(frame,f"Score: {overall}/100",
                (10,42),cv2.FONT_HERSHEY_SIMPLEX,0.65,sc_col,2,cv2.LINE_AA)

            out.write(frame)

    cap.release()
    out.release()
