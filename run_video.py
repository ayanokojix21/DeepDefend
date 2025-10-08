import argparse
import json
import os
import logging
from datetime import datetime
from extraction.media_extractor import MediaExtractor
from analysis.video_analyser import VideoAnalyzer


def summarize_and_save(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def setup_logger(log_dir: str, name: str = "deepdefend"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # avoid adding multiple handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger, log_path


def analyze_video(video_path: str, interval_duration: float = 2.0, frames_per_interval: int = 5):
    results_dir = os.path.abspath("results")
    logs_dir = os.path.join(results_dir, "logs")
    logger, log_path = setup_logger(logs_dir)

    logger.info(f"Starting analysis for: {video_path}")

    extractor = MediaExtractor(frames_per_interval=frames_per_interval)
    analyzer = VideoAnalyzer()

    timeline, video_info = extractor.extract_all(video_path, interval_duration=interval_duration)

    logger.info(f"Video duration: {video_info['duration']:.2f}s, fps: {video_info['fps']:.2f}, total_frames: {video_info['total_frames']}")

    all_interval_results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(timeline, desc="Processing intervals", unit="interval")
    except Exception:
        iterator = timeline

    for interval in iterator:
        logger.debug(f"Processing interval {interval['interval_id']} {interval['interval']}")
        try:
            interval_result = analyzer.analyze_interval(interval)
        except Exception as e:
            logger.exception(f"Error analyzing interval {interval.get('interval_id')}: {e}")
            interval_result = {
                'interval_id': interval['interval_id'],
                'interval': interval['interval'],
                'error': str(e)
            }
        all_interval_results.append(interval_result)

    summary = {
        'video_path': os.path.abspath(video_path),
        'analyzed_at': datetime.utcnow().isoformat() + 'Z',
        'video_info': video_info,
        'interval_results': all_interval_results
    }

    out_file = os.path.join(results_dir, os.path.splitext(os.path.basename(video_path))[0] + "_analysis.json")
    summarize_and_save(summary, out_file)

    logger.info(f"Analysis complete. Results saved to: {out_file}")
    logger.info(f"Log file: {log_path}")

    for r in all_interval_results:
        if 'error' in r:
            logger.warning(f"Interval {r.get('interval_id')} {r.get('interval')}: ERROR - {r['error']}")
            continue
        logger.info(f"Interval {r['interval_id']} {r['interval']}: fake_score={r.get('fake_score')}, confidence={r.get('confidence')}, face_detected={r.get('face_detected')}, suspicious_regions={r.get('suspicious_regions')}")


def main():
    parser = argparse.ArgumentParser(description="Run DeepDefend analysis on a video file")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Interval duration in seconds (default: 2.0)")
    parser.add_argument("--frames", "-f", type=int, default=5, help="Frames per interval to sample (default: 5)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return

    analyze_video(args.video, interval_duration=args.interval, frames_per_interval=args.frames)


if __name__ == "__main__":
    main()
