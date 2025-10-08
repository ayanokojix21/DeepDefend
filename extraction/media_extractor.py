import os
import cv2
import librosa
import subprocess
import numpy as np
import shutil
import logging
from typing import List, Dict, Tuple
from extraction.timeline_generator import TimelineGenerator

class MediaExtractor:
    
    def __init__(self, frames_per_interval: int = 5):
        self.frames_per_interval = frames_per_interval
    
    def get_video_info(self, video_path: str) -> Dict:
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }
    
    def extract_frames(self, video_path: str, timeline: List[Dict]) -> List[Dict]:
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        for interval in timeline:
            
            sample_times = np.linspace(
                interval['start'], 
                interval['end'], 
                self.frames_per_interval,
                endpoint=False
            )
            
            for sample_time in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                ret, frame = cap.read()
                
                if ret:
                    interval['video_data'].append({
                    'frame': frame,
                    'timestamp': round(sample_time, 2)
                })
        
        cap.release()
        
        return timeline
    
    def extract_audio(self, video_path: str, timeline: List[Dict]) -> List[Dict]:
        # Check if ffmpeg is available on PATH. If not, skip audio extraction gracefully.
        logger = logging.getLogger(__name__)
        if shutil.which('ffmpeg') is None:
            logger.warning('ffmpeg not found on PATH - skipping audio extraction. Install ffmpeg to enable audio processing.')
            for interval in timeline:
                interval['audio_data'] = {
                    'audio': np.zeros(16000 * 2),  
                    'sample_rate': 16000,
                    'has_audio': False
                }
            return timeline

        temp_audio = "temp_audio.wav"
        command = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', temp_audio
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        except subprocess.CalledProcessError:
            for interval in timeline:
                interval['audio_data'] = {
                    'audio': np.zeros(16000 * 2),  
                    'sample_rate': 16000,
                    'has_audio': False
                }
            return timeline
        
        audio, sr = librosa.load(temp_audio, sr=16000, mono=True)
        
        for interval in timeline:
            start_sample = int(interval['start'] * sr)
            end_sample = int(interval['end'] * sr)
            audio_chunk = audio[start_sample:end_sample]
            
            if len(audio_chunk) < sr * 0.5:
                audio_chunk = np.pad(audio_chunk, (0, int(sr * 0.5) - len(audio_chunk)))
            
            interval['audio_data'] = {
                'audio': audio_chunk,
                'sample_rate': sr,
                'has_audio': True
            }
        
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        return timeline
        
    
    def extract_all(self, video_path: str, interval_duration: float = 2.0) -> Tuple[List[Dict], Dict]:
        
        video_info = self.get_video_info(video_path)
        
        timeline_gen = TimelineGenerator(interval_duration)
        timeline = timeline_gen.create_timeline(video_info['duration'])
        
        self.extract_frames(video_path, timeline)
        self.extract_audio(video_path, timeline)
        
        return timeline, video_info