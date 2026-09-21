"""Pipeline stages, one module per stage.

Every stage module exposes ``process_video(video_dir, config)`` and is safe to
run on its own against an existing working directory, which is what makes the
pipeline restartable. Stage modules import their heavy dependencies at module
level, so import a stage only when you intend to run it.

    Stage 1   download    yt-dlp download of video + 16 kHz mono audio
    Stage 2   scenes      PySceneDetect shot boundaries
    Stage 3   audio       RMS normalisation and quality metrics
    Stage 4   transcribe  faster-whisper Russian transcription
    Stage 5A  align       global scenes + local segments (semantic embeddings)
    Stage 5B  translate   NLLB Russian -> English
    Stage 6A  intensity_ru  valence-arousal on Russian segments
    Stage 6B  intensity_en  valence-arousal on English translations
    Stage 7A  emotion_ru    7-class emotion on Russian segments
    Stage 7B  emotion_en    7-class emotion on English translations (ensemble)
    Stage 8   visualize   emotion timeline PNG
    Stage 9   export      consolidated CSV
"""

__all__ = [
    "align",
    "audio",
    "download",
    "emotion_en",
    "emotion_ru",
    "export",
    "intensity_en",
    "intensity_ru",
    "scenes",
    "transcribe",
    "translate",
    "visualize",
]
