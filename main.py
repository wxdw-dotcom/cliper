import os, json, uuid, asyncio, subprocess, re, tempfile, shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import anthropic
import yt_dlp

app = FastAPI(title="ClipViral Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/clipviral"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── In-memory job store ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ── Request models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    youtube_url: str
    anthropic_key: str = ""          # user can supply their own key
    num_clips: int = 5               # how many clips to generate
    max_duration: int = 60           # max seconds per clip
    niche: str = "umum"              # travel, kuliner, edukasi, dll

class ClipRequest(BaseModel):
    job_id: str
    clip_index: int                  # which clip to cut

# ── Helper: update job ───────────────────────────────────────────────────────
def job_update(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)

# ── Route: health ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Route: analyze YouTube video ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Antri...",
        "video_info": None,
        "clips": [],
        "error": None,
    }
    background_tasks.add_task(run_analyze, job_id, req)
    return {"job_id": job_id}

# ── Route: poll job status ───────────────────────────────────────────────────
@app.get("/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job tidak ditemukan")
    return jobs[job_id]

# ── Route: download a finished clip ─────────────────────────────────────────
@app.post("/clip")
async def get_clip(req: ClipRequest, background_tasks: BackgroundTasks):
    job = jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    if job["status"] != "done":
        raise HTTPException(400, "Analisis belum selesai")
    clips = job.get("clips", [])
    if req.clip_index >= len(clips):
        raise HTTPException(400, "Index clip tidak valid")

    clip_job_id = str(uuid.uuid4())
    jobs[clip_job_id] = {"status": "cutting", "progress": 0, "message": "Memotong video..."}
    background_tasks.add_task(run_cut, clip_job_id, req.job_id, req.clip_index)
    return {"clip_job_id": clip_job_id}

@app.get("/clip/{clip_job_id}/download")
def download_clip(clip_job_id: str):
    job = jobs.get(clip_job_id)
    if not job or job.get("status") != "ready":
        raise HTTPException(400, "Clip belum siap")
    path = job.get("file_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "File tidak ditemukan")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=Path(path).name,
        headers={"Content-Disposition": f"attachment; filename={Path(path).name}"}
    )

# ── Background: full analyze pipeline ────────────────────────────────────────
async def run_analyze(job_id: str, req: AnalyzeRequest):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Get video metadata (fast, no download)
        job_update(job_id, status="fetching_info", progress=5, message="Mengambil info video YouTube...")
        info = await get_video_info(req.youtube_url)
        job_update(job_id, video_info={
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "channel": info.get("uploader", ""),
            "view_count": info.get("view_count", 0),
        })

        # 2. Get transcript / subtitles
        job_update(job_id, progress=15, message="Mengambil transkrip video...")
        transcript = await get_transcript(req.youtube_url, job_dir)

        # 3. Ask Claude to find best clip moments
        job_update(job_id, progress=40, message="AI menganalisis momen terbaik...")
        api_key = req.anthropic_key or ANTHROPIC_API_KEY
        clips = await ai_find_clips(
            transcript=transcript,
            video_title=info.get("title", ""),
            duration=info.get("duration", 0),
            niche=req.niche,
            num_clips=req.num_clips,
            max_duration=req.max_duration,
            api_key=api_key,
        )

        # Store source url for later cutting
        job_update(job_id,
            status="done",
            progress=100,
            message="Analisis selesai!",
            clips=clips,
            source_url=req.youtube_url,
            job_dir=str(job_dir),
        )

    except Exception as e:
        job_update(job_id, status="error", error=str(e), message=f"Error: {e}")


async def run_cut(clip_job_id: str, parent_job_id: str, clip_index: int):
    parent = jobs.get(parent_job_id, {})
    clips = parent.get("clips", [])
    clip = clips[clip_index]
    source_url = parent.get("source_url", "")
    job_dir = Path(parent.get("job_dir", str(WORK_DIR / parent_job_id)))
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        jobs[clip_job_id].update(progress=10, message="Mendownload video dari YouTube...")
        video_path = await download_video(source_url, job_dir, clip["start"], clip["end"])

        jobs[clip_job_id].update(progress=70, message="Memotong clip...")
        out_path = job_dir / f"clip_{clip_index}_{uuid.uuid4().hex[:6]}.mp4"
        await ffmpeg_cut(video_path, str(out_path), clip["start"], clip["end"])

        jobs[clip_job_id].update(
            status="ready",
            progress=100,
            message="Clip siap didownload!",
            file_path=str(out_path),
        )
    except Exception as e:
        jobs[clip_job_id].update(status="error", message=f"Error: {e}")


# ── YouTube helpers ───────────────────────────────────────────────────────────
async def get_video_info(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
    loop = asyncio.get_event_loop()
    def _get():
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    return await loop.run_in_executor(None, _get)


async def get_transcript(url: str, job_dir: Path) -> str:
    """Try auto subtitles first, then manual, then generate from audio snippet."""
    loop = asyncio.get_event_loop()

    def _download_subs():
        opts = {
            "quiet": True,
            "no_warnings": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["id", "en"],
            "subtitlesformat": "vtt",
            "skip_download": True,
            "outtmpl": str(job_dir / "subs"),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    await loop.run_in_executor(None, _download_subs)

    # Find any .vtt file
    vtt_files = list(job_dir.glob("*.vtt"))
    if vtt_files:
        raw = vtt_files[0].read_text(errors="ignore")
        return parse_vtt_to_text(raw)

    # Fallback: return empty (AI will work with title + duration only)
    return ""


def parse_vtt_to_text(vtt: str) -> str:
    """Convert WebVTT to plain timestamped text for AI."""
    lines = vtt.split("\n")
    result = []
    time_pattern = re.compile(r"(\d+:\d+:\d+\.\d+|\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+|\d+:\d+\.\d+)")
    current_time = None
    current_text = []

    for line in lines:
        line = line.strip()
        m = time_pattern.match(line)
        if m:
            if current_time and current_text:
                result.append(f"[{current_time}] {' '.join(current_text)}")
            current_time = m.group(1)
            current_text = []
        elif line and not line.startswith("WEBVTT") and not line.isdigit():
            # Strip HTML tags from subtitles
            clean = re.sub(r"<[^>]+>", "", line)
            if clean:
                current_text.append(clean)

    if current_time and current_text:
        result.append(f"[{current_time}] {' '.join(current_text)}")

    return "\n".join(result[:500])  # limit to 500 lines for AI context


async def download_video(url: str, job_dir: Path, start: float, end: float) -> str:
    """Download only the needed segment using yt-dlp + ffmpeg."""
    loop = asyncio.get_event_loop()
    out_path = str(job_dir / f"source_{uuid.uuid4().hex[:8]}.mp4")

    def _dl():
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "outtmpl": out_path,
            "merge_output_format": "mp4",
            # Download only needed section for speed
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end + 5)]),
            "force_keyframes_at_cuts": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    await loop.run_in_executor(None, _dl)

    if not Path(out_path).exists():
        raise RuntimeError("Download gagal — file tidak ditemukan")
    return out_path


async def ffmpeg_cut(input_path: str, output_path: str, start: float, end: float):
    dur = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg gagal memotong video")


# ── AI: find best clip moments ───────────────────────────────────────────────
async def ai_find_clips(transcript: str, video_title: str, duration: int,
                         niche: str, num_clips: int, max_duration: int, api_key: str) -> list:

    if not api_key:
        raise ValueError("API Key Anthropic dibutuhkan untuk analisis AI")

    transcript_section = f"\n\nTRANSKRIP:\n{transcript}" if transcript else "\n\n(Transkrip tidak tersedia — analisis berdasarkan judul dan durasi)"

    prompt = f"""Kamu adalah ahli konten viral Indonesia. Analisis video YouTube ini dan temukan {num_clips} momen terbaik untuk dijadikan short clip viral.

JUDUL VIDEO: {video_title}
DURASI TOTAL: {duration} detik ({duration//60} menit {duration%60} detik)
NICHE: {niche}
MAKS DURASI CLIP: {max_duration} detik{transcript_section}

Tugas kamu:
1. Temukan {num_clips} momen yang paling berpotensi viral
2. Setiap clip harus punya hook kuat di detik pertama
3. Durasi ideal 15-{max_duration} detik
4. Pilih momen: reaksi kuat, info mengejutkan, konflik/drama, insight berharga, humor, atau momen emosional

Balas HANYA dengan JSON array ini, tanpa teks lain:
[
  {{
    "index": 0,
    "start": <detik mulai sebagai float>,
    "end": <detik selesai sebagai float>,
    "title": "<judul klip pendek>",
    "hook": "<kalimat hook pembuka 1-2 kalimat untuk dibacakan di 3 detik pertama>",
    "why_viral": "<alasan singkat kenapa momen ini berpotensi viral>",
    "viral_score": <angka 70-99>,
    "ctr_titles": [
      "<judul CTR tinggi untuk TikTok>",
      "<judul CTR tinggi untuk Instagram>",
      "<judul CTR tinggi untuk YouTube Shorts>"
    ],
    "hashtags": ["<hashtag1>", "<hashtag2>", "<hashtag3>", "<hashtag4>", "<hashtag5>"],
    "emotion": "<emosi utama: kejutan/tawa/inspirasi/penasaran/drama>"
  }}
]

Pastikan start dan end dalam detik (float), tidak melebihi durasi video ({duration} detik), dan setiap clip berdurasi 15-{max_duration} detik."""

    client = anthropic.Anthropic(api_key=api_key)
    loop = asyncio.get_event_loop()

    def _call():
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    raw = await loop.run_in_executor(None, _call)

    # Parse JSON
    raw = re.sub(r"```json|```", "", raw).strip()
    clips = json.loads(raw)

    # Validate and clamp
    valid = []
    for c in clips:
        start = max(0.0, float(c.get("start", 0)))
        end = min(float(duration), float(c.get("end", start + 30)))
        if end - start < 5:
            end = min(float(duration), start + 30)
        c["start"] = round(start, 2)
        c["end"] = round(end, 2)
        c["duration"] = round(end - start, 2)
        valid.append(c)

    return valid
