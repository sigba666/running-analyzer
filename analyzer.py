"""
analyzer.py – RunningAnalyzer v3 (무릎·허리·케이던스·수직진동)
"""
import cv2, math
import numpy as np
from scipy.signal import find_peaks, savgol_filter

_mp_import = __import__("mediapipe")
import mediapipe as mp

try:
    _mp_pose = mp.solutions.pose
except AttributeError:
    from mediapipe.python.solutions import pose as _mp_pose
_PL      = _mp_pose.PoseLandmark

_R = dict(shoulder=_PL.RIGHT_SHOULDER.value, hip=_PL.RIGHT_HIP.value,
          knee=_PL.RIGHT_KNEE.value,     ankle=_PL.RIGHT_ANKLE.value)
_L = dict(shoulder=_PL.LEFT_SHOULDER.value,  hip=_PL.LEFT_HIP.value,
          knee=_PL.LEFT_KNEE.value,      ankle=_PL.LEFT_ANKLE.value)

IDEAL_KNEE_ANGLE    = 160.0
IDEAL_TRUNK_ANGLE   =   5.0
IDEAL_CADENCE       = 175.0
IDEAL_V_OSC_CM      =   8.0
STABILITY_THRESHOLD =  20.0
ASYMMETRY_THRESHOLD =  15.0
ASSUMED_HEIGHT_CM   = 170.0


def _lm(lms, idx):    return (lms[idx].x, lms[idx].y)

def _angle_3pts(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
    cos = (ba[0]*bc[0]+ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc)+1e-9)
    return math.degrees(math.acos(max(-1.,min(1.,cos))))

def _trunk_angle(shoulder, hip):
    return math.degrees(math.atan2(abs(hip[0]-shoulder[0]), abs(hip[1]-shoulder[1])))

def _safe_savgol(arr, win=15, poly=2):
    n = len(arr)
    if n < 5: return np.array(arr)
    w = min(win, n if n%2==1 else n-1)
    if w < 3: w = 3
    if w%2==0: w -= 1
    return savgol_filter(arr, w, poly)

def _score_knee(e):      return max(0, round(100 - e*2.5))
def _score_trunk(e):     return max(0, round(100 - e*5))
def _score_cadence(c):
    if 170 <= c <= 180: return 100
    return max(0, round(100 - abs(c-175)*2))
def _score_v_osc(v):
    if 6 <= v <= 10: return 100
    return max(0, round(100 - abs(v-8)*10))
def _score_symmetry(a):  return max(0, round(100 - a*3))


def analyze_video(video_path: str, progress_cb=None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames=[]; r_knees=[]; l_knees=[]; r_trunks=[]; l_trunks=[]
    knee_asym=[]; trunk_asym=[]; frame_badness=[]
    r_ankle_y=[]; l_ankle_y=[]; hip_mid_y=[]; sho_mid_y=[]; ank_mid_y=[]

    with _mp_pose.Pose(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:
        for fi in range(total):
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if progress_cb: progress_cb(fi+1, total)
            if not res.pose_landmarks: continue
            lms = res.pose_landmarks.landmark

            rk = _angle_3pts(_lm(lms,_R['hip']),  _lm(lms,_R['knee']),  _lm(lms,_R['ankle']))
            lk = _angle_3pts(_lm(lms,_L['hip']),  _lm(lms,_L['knee']),  _lm(lms,_L['ankle']))
            rt = _trunk_angle(_lm(lms,_R['shoulder']), _lm(lms,_R['hip']))
            lt = _trunk_angle(_lm(lms,_L['shoulder']), _lm(lms,_L['hip']))

            r_ay=lms[_R['ankle']].y; l_ay=lms[_L['ankle']].y
            r_hy=lms[_R['hip']].y;   l_hy=lms[_L['hip']].y
            r_sy=lms[_R['shoulder']].y; l_sy=lms[_L['shoulder']].y

            knee_e  = (abs(rk-IDEAL_KNEE_ANGLE)+abs(lk-IDEAL_KNEE_ANGLE))/2
            trunk_e = (abs(rt-IDEAL_TRUNK_ANGLE)+abs(lt-IDEAL_TRUNK_ANGLE))/2
            asym_e  = abs(rk-lk)
            badness = 0.5*(knee_e/40)+0.3*(trunk_e/20)+0.2*(asym_e/30)

            frames.append(fi)
            r_knees.append(rk);  l_knees.append(lk)
            r_trunks.append(rt); l_trunks.append(lt)
            knee_asym.append(abs(rk-lk)); trunk_asym.append(abs(rt-lt))
            r_ankle_y.append(r_ay); l_ankle_y.append(l_ay)
            hip_mid_y.append((r_hy+l_hy)/2)
            sho_mid_y.append((r_sy+l_sy)/2)
            ank_mid_y.append((r_ay+l_ay)/2)
            frame_badness.append(round(float(badness),4))

    cap.release()
    if not frames: return {"error": "포즈를 감지하지 못했습니다."}

    rk_a=np.array(r_knees); lk_a=np.array(l_knees)
    rt_a=np.array(r_trunks); lt_a=np.array(l_trunks)
    ka_a=np.array(knee_asym); ta_a=np.array(trunk_asym)
    bd_a=np.array(frame_badness)

    duration_sec = len(frames) / fps
    min_dist = max(int(fps*0.25), 3)
    r_smooth = _safe_savgol(r_ankle_y)
    l_smooth = _safe_savgol(l_ankle_y)
    r_peaks,_ = find_peaks(r_smooth, distance=min_dist)
    l_peaks,_ = find_peaks(l_smooth, distance=min_dist)
    cadence = round((len(r_peaks)+len(l_peaks)) / duration_sec * 60, 1) if duration_sec > 0 else 0

    hip_arr=np.array(hip_mid_y); sho_arr=np.array(sho_mid_y); ank_arr=np.array(ank_mid_y)
    hip_smooth2 = _safe_savgol(hip_arr)
    osc_norm = np.percentile(hip_smooth2,95) - np.percentile(hip_smooth2,5)
    body_h_norm = float(np.mean(np.abs(ank_arr-sho_arr)))
    v_osc_cm = round(float((osc_norm/body_h_norm)*ASSUMED_HEIGHT_CM), 1) if body_h_norm > 1e-6 else 0.0

    rk_err=float(np.abs(rk_a-IDEAL_KNEE_ANGLE).mean())
    lk_err=float(np.abs(lk_a-IDEAL_KNEE_ANGLE).mean())
    rt_err=float(np.abs(rt_a-IDEAL_TRUNK_ANGLE).mean())
    lt_err=float(np.abs(lt_a-IDEAL_TRUNK_ANGLE).mean())
    avg_knee_err  = (rk_err+lk_err)/2
    avg_trunk_err = (rt_err+lt_err)/2

    scores = dict(
        knee    =_score_knee(avg_knee_err),
        trunk   =_score_trunk(avg_trunk_err),
        cadence =_score_cadence(cadence),
        v_osc   =_score_v_osc(v_osc_cm),
        symmetry=_score_symmetry(float(ka_a.mean())),
    )

    tips=[]
    if avg_knee_err  > 15: tips.append("무릎 신전 부족 → 보폭 줄이고 착지 자세 점검")
    if avg_trunk_err > 10: tips.append("허리 과도 기울어짐 → 코어 강화 및 상체 수직 유지")
    if cadence < 160:      tips.append(f"케이던스 낮음({cadence}spm) → 보폭 줄이고 회전수 늘리기")
    elif cadence > 200:    tips.append(f"케이던스 높음({cadence}spm) → 페이스 조절 필요")
    if v_osc_cm > 12:      tips.append(f"수직 진동 큼({v_osc_cm}cm) → 상하 바운싱 줄이기")
    elif v_osc_cm < 4:     tips.append(f"수직 진동 작음({v_osc_cm}cm) → 보폭이 너무 짧을 수 있음")
    if ka_a.mean() > ASYMMETRY_THRESHOLD: tips.append("좌우 무릎 비대칭 → 균형 훈련 필요")
    if rk_a.std()>STABILITY_THRESHOLD or lk_a.std()>STABILITY_THRESHOLD:
        tips.append("무릎 각도 변동 큼 → 균일한 보폭과 페이스 유지")

    step = max(1, len(frames)//300)
    bd_norm = (bd_a/bd_a.max()).tolist() if bd_a.max()>0 else bd_a.tolist()

    return dict(
        total_frames=total, valid_frames=len(frames),
        fps=fps, duration_sec=round(duration_sec,1),
        r_knee_mean=round(float(rk_a.mean()),1), r_knee_std=round(float(rk_a.std()),1),
        l_knee_mean=round(float(lk_a.mean()),1), l_knee_std=round(float(lk_a.std()),1),
        r_knee_error_mean=round(rk_err,1),        l_knee_error_mean=round(lk_err,1),
        r_trunk_mean=round(float(rt_a.mean()),1), r_trunk_std=round(float(rt_a.std()),1),
        l_trunk_mean=round(float(lt_a.mean()),1), l_trunk_std=round(float(lt_a.std()),1),
        r_trunk_error_mean=round(rt_err,1),        l_trunk_error_mean=round(lt_err,1),
        knee_asymmetry_mean=round(float(ka_a.mean()),1),
        trunk_asymmetry_mean=round(float(ta_a.mean()),1),
        knee_asymmetric=bool(ka_a.mean()>ASYMMETRY_THRESHOLD),
        trunk_asymmetric=bool(ta_a.mean()>ASYMMETRY_THRESHOLD),
        r_knee_unstable=bool(rk_a.std()>STABILITY_THRESHOLD),
        l_knee_unstable=bool(lk_a.std()>STABILITY_THRESHOLD),
        cadence=cadence,
        v_osc_cm=v_osc_cm,
        scores=scores,
        tips=tips,
        ideal=dict(knee_angle=IDEAL_KNEE_ANGLE, trunk_angle=IDEAL_TRUNK_ANGLE,
                   cadence=IDEAL_CADENCE, v_osc_cm=IDEAL_V_OSC_CM),
        chart=dict(
            frames    =frames[::step],
            r_knee    =[round(v,1) for v in r_knees[::step]],
            l_knee    =[round(v,1) for v in l_knees[::step]],
            r_trunk   =[round(v,1) for v in r_trunks[::step]],
            l_trunk   =[round(v,1) for v in l_trunks[::step]],
            knee_asym =[round(v,1) for v in list(ka_a)[::step]],
            trunk_asym=[round(v,1) for v in list(ta_a)[::step]],
            badness   =[round(v,3) for v in bd_norm[::step]],
            r_strikes =[frames[i] for i in r_peaks if i < len(frames)],
            l_strikes =[frames[i] for i in l_peaks if i < len(frames)],
        ),
    )
