import os
import cv2
import librosa
import subprocess
import numpy as np
import soundfile as sf
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'format': os.path.splitext(video_path)[1]
        }
    
    def extract_frames(self, video_path: str, timeline: List[Dict]) -> Dict[int, List[Dict]]:
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = {interval['interval_id']: [] for interval in timeline}
        
        for interval in timeline:
            interval_duration = interval['end'] - interval['start']
            
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
                    actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    results[interval['interval_id']].append({
                        'frame': frame.copy(),
                        'timestamp': round(actual_timestamp, 2),
                        'frame_number': frame_number,
                        'target_timestamp': round(sample_time, 2)
                    })
        
        cap.release()
        
        for interval in timeline:
            interval['video_data'] = results[interval['interval_id']]
        
        return results
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_audio.wav"
            
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  
            '-acodec', 'pcm_s16le',  
            '-ar', '16000',
            '-ac', '1',  
            '-y', 
            output_path
        ]
        
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction failed: {e}")
        
    def extract_audio_chunks(self, audio_path: str, timeline: List[Dict]) -> Dict[int, Dict]:
        
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        results = {}
        
        for interval in timeline:
            start_sample = int(interval['start'] * sr)
            end_sample = int(interval['end'] * sr)
            
            audio_chunk = audio[start_sample:end_sample]
            
            min_length = int(0.5 * sr)
            if len(audio_chunk) < min_length:
                audio_chunk = np.pad(
                    audio_chunk, 
                    (0, min_length - len(audio_chunk)),
                    mode='constant'
                )
            
            results[interval['interval_id']] = {
                'audio': audio_chunk,
                'sample_rate': sr,
                'duration': len(audio_chunk) / sr,
                'start': interval['start'],
                'end': interval['end'],
                'interval': interval['interval']
            }
        
        for interval in timeline:
            interval['audio_data'] = results[interval['interval_id']]
        
        return results
    
    def extract_all(self, video_path: str, interval_duration: float = 2.0):
        
        video_info = self.get_video_info(video_path)
        
        timeline_gen = TimelineGenerator(interval_duration)
        timeline = timeline_gen.create_timeline(video_info['duration'])
        
        self.extract_frames(video_path, timeline)
        
        audio_path = self.extract_audio(video_path)
        self.extract_audio_chunks(audio_path, timeline)
        
        return timeline, video_info