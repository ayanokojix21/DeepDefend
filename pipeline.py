import os
from typing import Dict
from extraction.media_extractor import MediaExtractor
from analysis.video_analyser import VideoAnalyzer
from analysis.audio_analyser import AudioAnalyzer
from analysis.llm_analyser import LLMFusion

class DeepfakeDetectionPipeline:
    """Complete deepfake detection pipeline"""
    
    def __init__(self):
        print("Initializing pipeline...")
        self.media_extractor = MediaExtractor(frames_per_interval=5)
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.llm_fusion = LLMFusion()
        print("Pipeline ready!")
    
    def analyze_video(self, video_path: str, interval_duration: float = 2.0) -> Dict:
        
        print(f"\n{'='*60}")
        print(f"DEEPFAKE DETECTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Video: {os.path.basename(video_path)}")
        
        print("\n[1/4] Extracting video frames and audio...")
        timeline, video_info = self.media_extractor.extract_all(video_path, interval_duration)
        print(f"✓ Extracted {len(timeline)} intervals ({video_info['duration']:.1f}s total)")
        
        print("\n[2/4] Analyzing video frames...")
        for i, interval in enumerate(timeline):
            print(f"  → Interval {i+1}/{len(timeline)}: {interval['interval']}")
            video_results = self.video_analyzer.analyze_interval(interval)
            interval['video_results'] = video_results
        print("✓ Video analysis complete")
        
        print("\n[3/4] Analyzing audio...")
        for i, interval in enumerate(timeline):
            print(f"  → Interval {i+1}/{len(timeline)}: {interval['interval']}")
            audio_results = self.audio_analyzer.analyze_interval(interval)
            interval['audio_results'] = audio_results
        print("✓ Audio analysis complete")
        
        print("\n[4/4] Generating AI-powered report...")
        final_report = self.llm_fusion.generate_report(timeline, video_info)
        print("✓ Report generated")
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Verdict: {final_report['verdict']}")
        print(f"Confidence: {final_report['confidence']}%")
        print(f"Overall Video Score: {final_report['overall_scores']['overall_video_score']}")
        print(f"Overall Audio Score: {final_report['overall_scores']['overall_audio_score']}")
        print(f"{'='*60}\n")
        
        return {
            'video_info': video_info,
            'timeline': timeline,
            'final_report': final_report,
            'summary': self.llm_fusion.generate_report(timeline, video_info)
        }


if __name__ == "__main__":
    
    pipeline = DeepfakeDetectionPipeline()
    
    video_path = "hack.mp4"
    results = pipeline.analyze_video(video_path, interval_duration=2.0)
    
    print("\nDetailed Analysis:")
    print(results['final_report']['detailed_analysis'])
    
    print("\nSuspicious Intervals:")
    for interval in results['final_report']['suspicious_intervals']:
        print(f"  • {interval['interval']} - Video: {interval['video_score']}, Audio: {interval['audio_score']}")
        print(f"    Video regions: {', '.join(interval['video_regions'])}")
        print(f"    Audio regions: {', '.join(interval['audio_regions'])}")
    
    json_report_path = "deepfake_report.json"
    with open(json_report_path, 'w') as f:
        import json
        json.dump(results['final_report'], f, indent=4)