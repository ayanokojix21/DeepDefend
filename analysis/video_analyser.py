import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from pytorch_grad_cam import GradCAM
from models.load_models import model_loader

class VideoAnalyzer:
    """Enhanced video analyzer with detailed facial region detection"""
    
    def __init__(self):
        self.model, self.processor = model_loader.load_video_model()
        self.device = model_loader.get_device()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_face(self, frame: np.ndarray) -> Dict:
        """Detect face and facial features"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = frame[y:y+h, x:x+w]
            
            gray_face = gray[y:y+h, x:x+w]
            
            return {
                'detected': True,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'face_crop': face_crop
            }
        
        return {
            'detected': False, 
            'bbox': None, 
            'face_crop': None
        }
    
    def predict_deepfake(self, frame: np.ndarray) -> Dict:
        """Predict if frame is deepfake"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        fake_prob = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
        confidence = max(probs[0]).item()
        
        return {
            'fake_score': round(fake_prob, 3),
            'confidence': round(confidence, 3),
            'label': 'fake' if fake_prob > 0.5 else 'real'
        }
    
    def get_gradcam_regions(self, frame: np.ndarray) -> Dict:
        """Generate Grad-CAM heatmap with detailed facial region analysis"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            
            pil_img = Image.fromarray(frame_rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            target_layers = [self.model.classifier[-1] if hasattr(self.model, 'classifier') 
                           else self.model.base_model.encoder.layers[-1]]
            
            cam = GradCAM(model=self.model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=inputs['pixel_values'], targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            suspicious_regions = self._identify_regions_enhanced(grayscale_cam)
            
            return {
                'regions': suspicious_regions,
                'heatmap_intensity': round(float(np.mean(grayscale_cam)), 3),
                'max_activation': round(float(np.max(grayscale_cam)), 3)
            }
        
        except Exception as e:
            return {
                'regions': ['Unable to generate heatmap'],
                'heatmap_intensity': 0.0,
                'max_activation': 0.0
            }
    
    def _identify_regions_enhanced(self, heatmap: np.ndarray, threshold: float = 0.55) -> List[str]:
        suspicious_regions = []
        h, w = heatmap.shape
        
        regions_map = {
            'forehead': heatmap[int(h*0.05):int(h*0.25), int(w*0.2):int(w*0.8)],
            'eyebrows': heatmap[int(h*0.25):int(h*0.35), int(w*0.15):int(w*0.85)],
            'left_eye': heatmap[int(h*0.30):int(h*0.45), int(w*0.15):int(w*0.40)],
            'right_eye': heatmap[int(h*0.30):int(h*0.45), int(w*0.60):int(w*0.85)],
            'eye_region': heatmap[int(h*0.28):int(h*0.48), int(w*0.15):int(w*0.85)],
            
            'nose_bridge': heatmap[int(h*0.35):int(h*0.50), int(w*0.40):int(w*0.60)],
            'nose': heatmap[int(h*0.45):int(h*0.60), int(w*0.38):int(w*0.62)],
            'left_cheek': heatmap[int(h*0.40):int(h*0.65), int(w*0.10):int(w*0.35)],
            'right_cheek': heatmap[int(h*0.40):int(h*0.65), int(w*0.65):int(w*0.90)],
            
            'upper_lip': heatmap[int(h*0.60):int(h*0.68), int(w*0.30):int(w*0.70)],
            'lower_lip': heatmap[int(h*0.68):int(h*0.75), int(w*0.32):int(w*0.68)],
            'mouth_area': heatmap[int(h*0.58):int(h*0.78), int(w*0.28):int(w*0.72)],
            'chin': heatmap[int(h*0.75):int(h*0.92), int(w*0.30):int(w*0.70)],
            
            'jaw_line': heatmap[int(h*0.50):int(h*0.90), int(w*0.05):int(w*0.95)],
            'face_boundary': heatmap[int(h*0.08):int(h*0.92), int(w*0.08):int(w*0.92)],
        }
        
        region_scores = {}
        for region_name, region_mask in regions_map.items():
            if region_mask.size > 0:
                avg_activation = np.mean(region_mask)
                max_activation = np.max(region_mask)
                std_activation = np.std(region_mask)
                
                region_scores[region_name] = {
                    'avg': avg_activation,
                    'max': max_activation,
                    'std': std_activation
                }
                
                if (avg_activation > threshold or 
                    max_activation > (threshold + 0.2) or 
                    std_activation > 0.25):
                    suspicious_regions.append(region_name)
        
        suspicious_regions.extend(self._analyze_texture_patterns(heatmap, threshold))
        suspicious_regions = list(dict.fromkeys(suspicious_regions))
        
        if len(suspicious_regions) == 0:
            return ['no_suspicious_regions_detected']
        
        return suspicious_regions
    
    def _analyze_texture_patterns(self, heatmap: np.ndarray, threshold: float) -> List[str]:
        patterns = []
        
        texture_variance = np.var(heatmap)
        if texture_variance < 0.02:
            patterns.append('overly_smooth_texture')
        
        if texture_variance > 0.15:
            patterns.append('high_frequency_artifacts')
        
        center = heatmap[int(heatmap.shape[0]*0.3):int(heatmap.shape[0]*0.7),
                        int(heatmap.shape[1]*0.3):int(heatmap.shape[1]*0.7)]
        edges = np.concatenate([
            heatmap[0:10, :].flatten(),
            heatmap[-10:, :].flatten(),
            heatmap[:, 0:10].flatten(),
            heatmap[:, -10:].flatten()
        ])
        
        center_mean = np.mean(center)
        edge_mean = np.mean(edges)
        
        if abs(center_mean - edge_mean) > 0.3:
            patterns.append('boundary_inconsistencies')
        
        left_half = heatmap[:, :heatmap.shape[1]//2]
        right_half = np.fliplr(heatmap[:, heatmap.shape[1]//2:])
        
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half - right_half))
        if symmetry_diff < 0.05:
            patterns.append('unnatural_facial_symmetry')
        
        return patterns
    
    def analyze_interval(self, interval_data: Dict) -> Dict:
        frames_data = interval_data['video_data']
        
        if not frames_data:
            return {
                'interval_id': interval_data['interval_id'],
                'interval': interval_data['interval'],
                'fake_score': 0.0,
                'confidence': 0.0,
                'suspicious_regions': [],
                'face_detected': False,
                'frame_results': []
            }
        
        frame_results = []
        total_fake_score = 0
        faces_detected = 0
        all_regions = set()
        region_frequency = {}  
        
        for frame_data in frames_data:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            face_info = self.detect_face(frame)
            
            if face_info['detected']:
                faces_detected += 1
                pred = self.predict_deepfake(face_info['face_crop'])
                regions = self.get_gradcam_regions(face_info['face_crop'])
            else:
                pred = self.predict_deepfake(frame)
                regions = {
                    'regions': ['no_face_detected'], 
                    'heatmap_intensity': 0.0,
                    'max_activation': 0.0
                }
            
            total_fake_score += pred['fake_score']
            all_regions.update(regions['regions'])
            
            for region in regions['regions']:
                region_frequency[region] = region_frequency.get(region, 0) + 1
            
            frame_results.append({
                'timestamp': timestamp,
                'fake_score': pred['fake_score'],
                'confidence': pred['confidence'],
                'face_detected': face_info['detected'],
                'regions': regions['regions'],
                'heatmap_intensity': regions.get('heatmap_intensity', 0.0)
            })
        
        avg_fake_score = total_fake_score / len(frames_data)
        
        threshold_count = len(frames_data) * 0.4
        consistent_suspicious_regions = [
            region for region, count in region_frequency.items() 
            if count >= threshold_count and region != 'no_suspicious_regions_detected'
        ]
        
        return {
            'interval_id': interval_data['interval_id'],
            'interval': interval_data['interval'],
            'fake_score': round(avg_fake_score, 3),
            'confidence': round(np.mean([f['confidence'] for f in frame_results]), 3),
            'suspicious_regions': consistent_suspicious_regions if consistent_suspicious_regions else list(all_regions),
            'face_detected': faces_detected > 0,
            'frame_count': len(frames_data),
            'frames_with_faces': faces_detected,
            'avg_heatmap_intensity': round(np.mean([f.get('heatmap_intensity', 0) for f in frame_results]), 3),
            'frame_results': frame_results,
            'region_frequency': region_frequency 
        }