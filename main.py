"""
main.py – FastAPI 러닝 자세 분석 서버

실행:
    pip install fastapi uvicorn python-multipart mediapipe opencv-python-headless
    uvicorn main:app --reload --port 8000

접속: http://localhost:8000
"""
import uuid
import shutil
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_video

# ── 디렉토리 설정 ────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── 분석 작업 상태 저장 (간단한 인메모리 dict) ───────────────
jobs: Dict[str, dict] = {}   # job_id -> { status, progress, result }

app = FastAPI(title="러닝 자세 분석 API")

# ── 정적 파일 (프론트엔드 HTML) ──────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 메인 페이지 ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html = (Path("static") / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── 영상 업로드 & 분석 시작 ──────────────────────────────────
@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # 확장자 체크
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "mp4 / mov / avi / mkv 형식만 업로드 가능합니다.")

    job_id   = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    # 파일 저장
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 작업 등록
    jobs[job_id] = {"status": "processing", "progress": 0, "total": 0, "result": None}

    # 백그라운드 분석
    background_tasks.add_task(_run_analysis, job_id, str(save_path))

    return {"job_id": job_id}


# ── 진행률 조회 ──────────────────────────────────────────────
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job_id를 찾을 수 없습니다.")
    return job


# ── 분석 결과 조회 ───────────────────────────────────────────
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job_id를 찾을 수 없습니다.")
    if job["status"] != "done":
        raise HTTPException(202, "아직 분석 중입니다.")
    return JSONResponse(job["result"])


# ── 백그라운드 분석 함수 ─────────────────────────────────────
def _run_analysis(job_id: str, video_path: str):
    def progress_cb(current, total):
        jobs[job_id]["progress"] = current
        jobs[job_id]["total"]    = total

    try:
        result = analyze_video(video_path, progress_cb=progress_cb)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["result"] = {"error": str(e)}
    finally:
        # 업로드 파일 삭제
        Path(video_path).unlink(missing_ok=True)
